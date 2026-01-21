# asr/adapter.py
"""
ASR Adapter Pattern

Provides a unified interface for different ASR engines to work with SegmentManager.
Each adapter normalizes its engine's output to the standard Hypothesis format.

Standard Hypothesis Format (expected by SegmentManager):
{
    "type": "hypothesis",
    "segment_id": str,
    "revision": int,
    "text": str,
    "words": [
        {
            "word": str,
            "start_ms": int,
            "end_ms": int,
            "probability": float,
        },
        ...
    ],
    "confidence": float,
    "start_time_ms": int,  # buffer start
    "end_time_ms": int,    # buffer end
}

Usage:
    from fap.asr.adapter import create_asr_adapter
    
    # Create adapter (auto-detects engine type)
    adapter = create_asr_adapter("whisper", model_size="medium")
    # or
    adapter = create_asr_adapter("google", language_code="en-US")
    
    # Feed audio - returns normalized hypothesis
    hypothesis = adapter.feed(audio_bytes)
    if hypothesis:
        segment_output = segment_manager.ingest(hypothesis)
"""

from abc import ABC, abstractmethod
from typing import Any
import time


class ASRAdapter(ABC):
    """
    Abstract base class for ASR adapters.
    
    Adapters wrap ASR engines and normalize their output to the standard
    Hypothesis format expected by SegmentManager.
    """
    
    @abstractmethod
    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio chunk and return normalized hypothesis.
        
        Args:
            audio: PCM16 audio bytes (16kHz, mono)
            
        Returns:
            Normalized hypothesis dict or None if no transcription available
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the adapter and underlying engine state."""
        pass
    
    @abstractmethod
    def finalize(self) -> dict | None:
        """
        Finalize the current segment and return any remaining hypothesis.
        Called when stream ends or segment boundary detected.
        
        Returns:
            Final hypothesis dict or None
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'whisper', 'google')."""
        pass
    
    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        """Return whether the adapter is currently streaming."""
        pass


class WhisperAdapter(ASRAdapter):
    """
    Adapter for faster-whisper ASR engine.
    
    Whisper uses a rolling buffer approach - it re-transcribes
    the entire buffer on each call, providing word-level timestamps.
    """
    
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        model: Any = None,
        buffer_duration_ms: int = 2000,
    ):
        """
        Initialize Whisper adapter.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (auto, cpu, cuda)
            model: Pre-loaded WhisperModel instance (optional)
            buffer_duration_ms: Rolling buffer duration
        """
        from .whisper_engine import WhisperASR
        
        self._engine = WhisperASR(
            model_size=model_size,
            device=device,
            model=model,
            buffer_duration_ms=buffer_duration_ms,
        )
        self._is_streaming = False
        self._last_hypothesis: dict | None = None
    
    @property
    def provider_name(self) -> str:
        return "whisper"
    
    @property
    def is_streaming(self) -> bool:
        return self._is_streaming
    
    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio to Whisper and return normalized hypothesis.
        
        Whisper already outputs in the expected format, so minimal
        normalization is needed.
        """
        self._is_streaming = True
        
        hypothesis = self._engine.feed(audio)
        
        if hypothesis:
            # Whisper output is already in the correct format
            # Just ensure all required fields are present
            normalized = self._normalize_hypothesis(hypothesis)
            self._last_hypothesis = normalized
            return normalized
        
        return None
    
    def _normalize_hypothesis(self, hypothesis: dict) -> dict:
        """Ensure hypothesis has all required fields."""
        return {
            "type": hypothesis.get("type", "hypothesis"),
            "segment_id": hypothesis.get("segment_id", f"whisper-{int(time.time() * 1000)}"),
            "revision": hypothesis.get("revision", 0),
            "text": hypothesis.get("text", ""),
            "words": hypothesis.get("words", []),
            "confidence": hypothesis.get("confidence", 0.0),
            "start_time_ms": hypothesis.get("start_time_ms", 0),
            "end_time_ms": hypothesis.get("end_time_ms", 0),
        }
    
    def reset(self) -> None:
        """Reset Whisper engine state."""
        self._engine.reset()
        self._is_streaming = False
        self._last_hypothesis = None
    
    def finalize(self) -> dict | None:
        """
        Finalize current segment.
        
        For Whisper, we return the last hypothesis as final.
        """
        self._is_streaming = False
        
        if self._last_hypothesis:
            # Mark as final - use is_final (standardized flag for SegmentManager)
            final_hypothesis = self._last_hypothesis.copy()
            final_hypothesis["is_final"] = True
            self._last_hypothesis = None
            return final_hypothesis
        
        return None


class GoogleAdapter(ASRAdapter):
    """
    Adapter for Google Cloud Speech-to-Text ASR engine.
    
    Google uses true streaming with hypothesis rewriting.
    
    Key insight: Engine's feed() returns one result and puts audio in queue.
    We need to also drain any additional results that accumulated.
    Final results should be returned before interims.
    """
    
    def __init__(
        self,
        language_code: str = "en-US",
        sample_rate_hz: int = 16000,
        credentials_path: str | None = None,
        model: str = "latest_long",
    ):
        from .google_engine import GoogleCloudASR
        import queue as queue_module
        self._queue_module = queue_module
        
        self._engine = GoogleCloudASR(
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
            credentials_path=credentials_path,
            model=model,
        )
        self._is_streaming = False
        self._last_hypothesis: dict | None = None
        self._revision = 0
        
        # Queue final results that need to be returned
        self._pending_finals: list[dict] = []
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    @property
    def is_streaming(self) -> bool:
        return self._is_streaming
    
    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio to Google Cloud and return normalized hypothesis.
        
        Strategy:
        1. Return pending finals first (from previous call)
        2. Check engine's finals_queue for any finals we might have missed
        3. Feed audio to engine
        4. Check finals_queue again
        5. Drain regular response queue for latest interim
        6. Return: pending final > latest interim > None
        
        Key insight: Finals go to a separate queue so they can't be lost
        even if interims arrive between feed() calls.
        """
        self._is_streaming = True
        
        # Step 1: Return pending finals from previous call
        if self._pending_finals:
            result = self._pending_finals.pop(0)
            self._last_hypothesis = result
            return result
        
        # Step 2: Check for finals that arrived since last call
        self._drain_finals_queue()
        if self._pending_finals:
            result = self._pending_finals.pop(0)
            self._last_hypothesis = result
            return result
        
        # Step 3: Feed audio to engine
        self._engine.feed(audio)
        
        # Step 4: Small delay then check finals again
        import time as time_mod
        time_mod.sleep(0.02)  # 20ms
        
        self._drain_finals_queue()
        if self._pending_finals:
            result = self._pending_finals.pop(0)
            self._last_hypothesis = result
            return result
        
        # Step 5: No finals - get latest interim from response queue
        latest_interim = None
        while True:
            try:
                result = self._engine._response_queue.get_nowait()
                if result and not result.get("is_final_from_google", False):
                    latest_interim = result
                # Skip finals in response queue (already in finals_queue)
            except self._queue_module.Empty:
                break
        
        # Step 6: Return latest interim
        if latest_interim:
            self._revision += 1
            result = self._normalize_hypothesis(latest_interim, is_final=False)
            self._last_hypothesis = result
            return result
        
        return None
    
    def _drain_finals_queue(self):
        """Drain all finals from engine's finals queue into pending_finals."""
        while True:
            try:
                final = self._engine._finals_queue.get_nowait()
                if final:
                    self._revision += 1
                    normalized = self._normalize_hypothesis(final, is_final=True)
                    self._pending_finals.append(normalized)
                    print(f"   📬 Captured final: \"{final.get('text', '')[:50]}...\"")
            except self._queue_module.Empty:
                break
    
    def _normalize_hypothesis(self, hypothesis: dict, is_final: bool) -> dict:
        """Normalize hypothesis to standard format."""
        import time
        return {
            "type": "hypothesis",
            "segment_id": hypothesis.get("segment_id", f"google-{int(time.time() * 1000)}"),
            "revision": self._revision,
            "text": hypothesis.get("text", ""),
            "words": hypothesis.get("words", []),
            "confidence": hypothesis.get("confidence", 0.9),
            "start_time_ms": hypothesis.get("start_time_ms", 0),
            "end_time_ms": hypothesis.get("end_time_ms", 0),
            "is_final": is_final,
        }
    
    def reset(self) -> None:
        """Reset Google engine state and stop streaming."""
        self._stop_streaming()
        self._engine.reset()
        self._is_streaming = False
        self._last_hypothesis = None
        self._revision = 0
        self._pending_finals = []
    
    def _stop_streaming(self) -> None:
        """Stop the Google streaming thread gracefully."""
        try:
            # Signal the engine to stop
            self._engine._is_streaming = False
            # Send poison pill to unblock the audio generator
            self._engine._audio_queue.put(None)
        except:
            pass
    
    def finalize(self) -> dict | None:
        """
        Finalize current segment and stop streaming.
        
        Important: We may have already processed all finals during feed().
        Only return a final here if there's a pending one we haven't returned yet.
        """
        import time as time_module
        
        self._is_streaming = False
        
        # Check if we already have pending finals (shouldn't re-process)
        if self._pending_finals:
            self._stop_streaming()
            return self._pending_finals.pop(0)
        
        # Send silence frame to nudge Google into finalization
        try:
            self._engine._audio_queue.put(b"\x00" * 640)  # 20ms of silence
        except:
            pass
        
        # Brief wait to see if a NEW final arrives
        deadline = time_module.time() + 0.3
        while time_module.time() < deadline:
            self._drain_finals_queue()
            if self._pending_finals:
                self._stop_streaming()
                return self._pending_finals.pop(0)
            time_module.sleep(0.05)
        
        # Stop streaming before returning
        self._stop_streaming()
        
        # No new finals - just return None (don't fabricate a final from last_hypothesis
        # as that would cause duplication if it was already processed)
        return None

class OpenAIAdapter(ASRAdapter):
    """
    Adapter for OpenAI Realtime transcription API.

    Uses gpt-4o-transcribe model with streaming WebSocket connection.
    No word-level timestamps, so uses "rewriting" mode with time-based stability.

    Key insight: Engine's feed() returns one result and puts audio in queue.
    We need to also drain any additional results that accumulated.
    Final results should be returned before interims.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-transcribe",
        language: str = "en",
    ):
        from .openai_engine import OpenAIRealtimeASR
        import queue as queue_module
        self._queue_module = queue_module

        self._engine = OpenAIRealtimeASR(
            api_key=api_key,
            model=model,
            language=language,
        )
        self._is_streaming = False
        self._last_hypothesis: dict | None = None
        self._revision = 0

        # Queue final results that need to be returned
        self._pending_finals: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio to OpenAI and return normalized hypothesis.

        Strategy (same as GoogleAdapter):
        1. Return pending finals first (from previous call)
        2. Check engine's finals_queue for any finals we might have missed
        3. Feed audio to engine
        4. Check finals_queue again
        5. Drain regular response queue for latest interim
        6. Return: pending final > latest interim > None
        """
        self._is_streaming = True

        # Step 1: Return pending finals from previous call
        if self._pending_finals:
            result = self._pending_finals.pop(0)
            self._last_hypothesis = result
            return result

        # Step 2: Check for finals that arrived since last call
        self._drain_finals_queue()
        if self._pending_finals:
            result = self._pending_finals.pop(0)
            self._last_hypothesis = result
            return result

        # Step 3: Feed audio to engine
        self._engine.feed(audio)

        # Step 4: Brief check for finals (no blocking sleep for lower latency)
        self._drain_finals_queue()
        if self._pending_finals:
            result = self._pending_finals.pop(0)
            self._last_hypothesis = result
            return result

        # Step 5: No finals - get latest interim from response queue
        latest_interim = None
        while True:
            try:
                result = self._engine._response_queue.get_nowait()
                if result and not result.get("is_final_from_openai", False):
                    latest_interim = result
                # Skip finals in response queue (already in finals_queue)
            except self._queue_module.Empty:
                break

        # Step 6: Return latest interim
        if latest_interim:
            self._revision += 1
            result = self._normalize_hypothesis(latest_interim, is_final=False)
            self._last_hypothesis = result
            return result

        return None

    def _drain_finals_queue(self):
        """Drain all finals from engine's finals queue into pending_finals."""
        while True:
            try:
                final = self._engine._finals_queue.get_nowait()
                if final:
                    self._revision += 1
                    normalized = self._normalize_hypothesis(final, is_final=True)
                    self._pending_finals.append(normalized)
                    print(f"   📬 Captured OpenAI final: \"{final.get('text', '')[:50]}...\"")
            except self._queue_module.Empty:
                break

    def _normalize_hypothesis(self, hypothesis: dict, is_final: bool) -> dict:
        """Normalize hypothesis to standard format."""
        import time
        return {
            "type": "hypothesis",
            "segment_id": hypothesis.get("segment_id", f"openai-{int(time.time() * 1000)}"),
            "revision": self._revision,
            "text": hypothesis.get("text", ""),
            "words": hypothesis.get("words", []),
            "confidence": hypothesis.get("confidence", 0.9),
            "start_time_ms": hypothesis.get("start_time_ms", 0),
            "end_time_ms": hypothesis.get("end_time_ms", 0),
            "is_final": is_final,
        }

    def reset(self) -> None:
        """Reset OpenAI engine state and stop streaming."""
        self._stop_streaming()
        self._engine.reset()
        self._is_streaming = False
        self._last_hypothesis = None
        self._revision = 0
        self._pending_finals = []

    def _stop_streaming(self) -> None:
        """Stop the OpenAI streaming thread gracefully."""
        try:
            # Signal the engine to stop
            self._engine._is_streaming = False
            # Send poison pill to unblock the audio generator
            self._engine._audio_queue.put(None)
        except:
            pass

    def finalize(self) -> dict | None:
        """
        Finalize current segment and stop streaming.

        Important: We may have already processed all finals during feed().
        Only return a final here if there's a pending one we haven't returned yet.
        """
        import time as time_module

        self._is_streaming = False

        # Check if we already have pending finals
        if self._pending_finals:
            self._stop_streaming()
            return self._pending_finals.pop(0)

        # Brief wait to see if a final arrives
        deadline = time_module.time() + 0.3
        while time_module.time() < deadline:
            self._drain_finals_queue()
            if self._pending_finals:
                self._stop_streaming()
                return self._pending_finals.pop(0)
            time_module.sleep(0.05)

        # Stop streaming before returning
        self._stop_streaming()

        # No new finals - return None
        return None


def create_asr_adapter(
    provider: str = "whisper",
    **kwargs: Any,
) -> ASRAdapter:
    """
    Factory function to create an ASR adapter.
    
    Args:
        provider: ASR provider ("whisper" or "google")
        **kwargs: Provider-specific arguments
        
    Returns:
        ASRAdapter instance
        
    Examples:
        # Whisper adapter
        adapter = create_asr_adapter("whisper", model_size="medium")
        
        # Google adapter
        adapter = create_asr_adapter("google", language_code="ko-KR")
        
        # With pre-loaded model
        adapter = create_asr_adapter("whisper", model=shared_model)
    """
    import os
    
    # Get provider from env if not specified
    if provider is None:
        provider = os.getenv("ASR_PROVIDER", "whisper").lower()
    
    print(f"🔧 Creating ASR adapter: {provider}")
    
    if provider == "whisper":
        # Default kwargs for whisper
        whisper_defaults = {
            "model_size": os.getenv("WHISPER_MODEL_SIZE", "medium"),
            "device": os.getenv("WHISPER_DEVICE", "auto"),
        }
        whisper_defaults.update(kwargs)
        return WhisperAdapter(**whisper_defaults)
    
    elif provider == "google":
        # Default kwargs for google
        google_defaults = {
            "language_code": os.getenv("GOOGLE_ASR_LANGUAGE", "en-US"),
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            "model": os.getenv("GOOGLE_ASR_MODEL", "latest_long"),
        }
        google_defaults.update(kwargs)
        return GoogleAdapter(**google_defaults)

    elif provider == "openai":
        # Default kwargs for openai
        openai_defaults = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_ASR_MODEL", "gpt-4o-transcribe"),
            "language": os.getenv("OPENAI_ASR_LANGUAGE", "en"),
        }
        openai_defaults.update(kwargs)
        return OpenAIAdapter(**openai_defaults)

    else:
        raise ValueError(
            f"Unknown ASR provider: {provider}. "
            f"Supported providers: whisper, google, openai"
        )