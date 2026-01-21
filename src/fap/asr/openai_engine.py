# asr/openai_engine.py
"""
OpenAI Realtime API ASR Engine implementation.

Uses the OpenAI Realtime API with WebSocket for streaming transcription
using the gpt-4o-transcribe model.

API Reference: https://platform.openai.com/docs/guides/realtime-transcription
"""

import os
import time
import json
import base64
import queue
import threading
from typing import Any

from .base import ASREngine, Hypothesis, WordInfo
from .utils import resample_audio


class OpenAIRealtimeASR(ASREngine):
    """
    OpenAI Realtime API ASR engine using gpt-4o-transcribe.

    Uses WebSocket connection to OpenAI's Realtime API for streaming
    transcription. Handles audio resampling from 16kHz to 24kHz.
    """

    WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
    TARGET_SAMPLE_RATE = 24000  # OpenAI requires 24kHz

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-transcribe",
        language: str = "en",
        input_sample_rate: int = 16000,
    ):
        """
        Initialize OpenAI Realtime ASR engine.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-transcribe)
            language: Language code (default: en)
            input_sample_rate: Input audio sample rate (default: 16000)
        """
        print(f"🌐 Initializing OpenAI Realtime ASR: model={model}, language={language}")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.language = language
        self.input_sample_rate = input_sample_rate

        # State
        self.segment_id = f"openai-seg-{int(time.time() * 1000)}"
        self.revision = 0
        self.session_start_time_ms = int(time.time() * 1000)

        # Streaming state
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._response_queue: queue.Queue[Hypothesis | None] = queue.Queue()
        self._finals_queue: queue.Queue[Hypothesis | None] = queue.Queue()
        self._streaming_thread: threading.Thread | None = None
        self._is_streaming = False

        # Track current transcript for delta accumulation
        self._current_transcript = ""
        self._audio_buffer_ms = 0  # Track how much audio we've sent

        # Lock for thread safety
        self._lock = threading.Lock()

        print("✅ OpenAI Realtime ASR initialized")

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_current_time_ms(self) -> int:
        """Get current time in milliseconds."""
        return int(time.time() * 1000)

    def _stream_thread(self):
        """Background thread for WebSocket communication with OpenAI."""
        import websockets.sync.client as ws_client

        try:
            # Connect to OpenAI Realtime API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            }

            print(f"🔌 Connecting to OpenAI Realtime API...")

            with ws_client.connect(
                self.WEBSOCKET_URL,
                additional_headers=headers,
            ) as websocket:
                print(f"✅ Connected to OpenAI Realtime API")

                # Configure transcription session
                # VAD settings tuned for better word capture:
                # - threshold: 0.2 (very sensitive to speech)
                # - prefix_padding_ms: 1000 (capture 1s of audio before VAD triggers)
                # - silence_duration_ms: 500 (end turn after 500ms silence)
                session_config = {
                    "type": "transcription_session.update",
                    "session": {
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": self.model,
                            "language": self.language,
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.2,  # Very sensitive - detects speech start earlier
                            "prefix_padding_ms": 1000,  # Buffer 1s before VAD triggers
                            "silence_duration_ms": 500,  # Wait for pause before ending turn
                        },
                    },
                }
                websocket.send(json.dumps(session_config))
                print(f"📤 Sent session config: model={self.model}, language={self.language}, VAD threshold=0.2, prefix_padding=1000ms")

                # Start receiver thread
                receiver_thread = threading.Thread(
                    target=self._receive_events,
                    args=(websocket,),
                    daemon=True,
                )
                receiver_thread.start()

                # Send audio loop
                while self._is_streaming:
                    try:
                        audio_chunk = self._audio_queue.get(timeout=0.05)  # 50ms for lower latency
                        if audio_chunk is None:  # Poison pill
                            break

                        # Resample from 16kHz to 24kHz
                        resampled = resample_audio(
                            audio_chunk,
                            self.input_sample_rate,
                            self.TARGET_SAMPLE_RATE,
                        )

                        # Track audio duration
                        chunk_duration_ms = (len(audio_chunk) // 2) * 1000 // self.input_sample_rate
                        self._audio_buffer_ms += chunk_duration_ms

                        # Send audio
                        audio_event = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(resampled).decode("utf-8"),
                        }
                        websocket.send(json.dumps(audio_event))

                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"❌ Error sending audio: {e}")
                        break

                # Send commit to finalize any pending audio
                try:
                    websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                except Exception:
                    pass

        except Exception as e:
            print(f"❌ OpenAI Realtime streaming error: {e}")
        finally:
            self._is_streaming = False
            print("🛑 OpenAI streaming thread ended")

    def _receive_events(self, websocket: Any):
        """Receive and process events from OpenAI WebSocket."""
        try:
            while self._is_streaming:
                try:
                    message = websocket.recv(timeout=0.05)  # 50ms for lower latency
                    event = json.loads(message)
                    self._handle_event(event)
                except TimeoutError:
                    continue
                except Exception as e:
                    if self._is_streaming:
                        print(f"❌ Error receiving event: {e}")
                    break
        except Exception as e:
            print(f"❌ Receiver thread error: {e}")

    def _handle_event(self, event: dict):
        """Handle a single event from OpenAI."""
        event_type = event.get("type", "")

        if event_type == "transcription_session.created":
            session = event.get("session", {})
            turn_detection = session.get("turn_detection", {})
            print(f"📝 Transcription session created")
            print(f"   Turn detection: {turn_detection.get('type')}, threshold={turn_detection.get('threshold')}, prefix={turn_detection.get('prefix_padding_ms')}ms")

        elif event_type == "transcription_session.updated":
            session = event.get("session", {})
            turn_detection = session.get("turn_detection", {})
            print(f"📝 Transcription session updated")
            print(f"   Turn detection: {turn_detection.get('type')}, threshold={turn_detection.get('threshold')}, prefix={turn_detection.get('prefix_padding_ms')}ms")

        elif event_type == "conversation.item.input_audio_transcription.delta":
            # Incremental transcript update
            delta = event.get("delta", "")
            if delta:
                self._current_transcript += delta
                self._emit_hypothesis(is_final=False)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # Final transcript for this segment
            transcript = event.get("transcript", "")
            if transcript:
                self._current_transcript = transcript
                self._emit_hypothesis(is_final=True)

                # Reset for next segment
                self._current_transcript = ""
                self.segment_id = f"openai-seg-{int(time.time() * 1000)}"

        elif event_type == "input_audio_buffer.speech_started":
            audio_start_ms = event.get("audio_start_ms", "unknown")
            item_id = event.get("item_id", "unknown")
            print(f"🎤 Speech started at {audio_start_ms}ms (item: {item_id})")

        elif event_type == "input_audio_buffer.speech_stopped":
            audio_end_ms = event.get("audio_end_ms", "unknown")
            item_id = event.get("item_id", "unknown")
            print(f"🔇 Speech stopped at {audio_end_ms}ms (item: {item_id})")

        elif event_type == "input_audio_buffer.committed":
            print("📦 Audio buffer committed")

        elif event_type == "error":
            error = event.get("error", {})
            print(f"❌ OpenAI error: {error.get('message', 'Unknown error')}")

    def _emit_hypothesis(self, is_final: bool):
        """Create and emit a hypothesis from current transcript."""
        if not self._current_transcript.strip():
            return

        self.revision += 1

        # Create approximate word timestamps (no word-level timestamps from API)
        words_with_timestamps: list[WordInfo] = []
        words = self._current_transcript.split()

        if words:
            # Estimate word duration based on buffer position
            word_duration_ms = 200  # Approximate 200ms per word
            total_duration = len(words) * word_duration_ms
            start_ms = self.session_start_time_ms + max(0, self._audio_buffer_ms - total_duration)

            for word in words:
                words_with_timestamps.append({
                    "word": word,
                    "start_ms": start_ms,
                    "end_ms": start_ms + word_duration_ms,
                    "probability": 0.9,  # OpenAI doesn't provide confidence
                })
                start_ms += word_duration_ms

        hypothesis: Hypothesis = {
            "type": "hypothesis",
            "segment_id": self.segment_id,
            "revision": self.revision,
            "text": self._current_transcript,
            "words": words_with_timestamps,
            "confidence": 0.9,
            "start_time_ms": self.session_start_time_ms,
            "end_time_ms": self.session_start_time_ms + self._audio_buffer_ms,
            "is_final_from_openai": is_final,
        }

        # Log
        prefix = "✅" if is_final else "⏳"
        print(f"{prefix} OpenAI ASR (rev {self.revision}): \"{self._current_transcript}\"")

        # Put in queues
        self._response_queue.put(hypothesis)
        if is_final:
            self._finals_queue.put(hypothesis)

    def _start_streaming(self):
        """Start the streaming thread if not already running."""
        if not self._is_streaming:
            self._is_streaming = True
            self.session_start_time_ms = self._get_current_time_ms()
            self._streaming_thread = threading.Thread(
                target=self._stream_thread,
                daemon=True,
            )
            self._streaming_thread.start()
            print("🎙️ OpenAI Realtime streaming started")

    def feed(self, audio: bytes) -> Hypothesis | None:
        """
        Feed audio chunk and return hypothesis if available.

        Args:
            audio: PCM16 audio bytes (16kHz)

        Returns:
            Hypothesis dict if transcription available, None otherwise
        """
        # Start streaming on first audio
        if not self._is_streaming:
            self._start_streaming()

        # Add audio to queue
        self._audio_queue.put(audio)

        # Return latest hypothesis if available (non-blocking)
        try:
            return self._response_queue.get_nowait()
        except queue.Empty:
            return None

    def reset(self) -> None:
        """Reset the engine state for a new segment/session."""
        # Stop current stream
        self._is_streaming = False
        self._audio_queue.put(None)

        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)

        # Clear queues
        for q in [self._audio_queue, self._response_queue, self._finals_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Reset state
        self._audio_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._finals_queue = queue.Queue()
        self.segment_id = f"openai-seg-{int(time.time() * 1000)}"
        self.revision = 0
        self.session_start_time_ms = int(time.time() * 1000)
        self._current_transcript = ""
        self._audio_buffer_ms = 0

        print("🔄 OpenAI Realtime ASR reset (ready for new session)")

    def __del__(self):
        """Cleanup on destruction."""
        self._is_streaming = False
        try:
            self._audio_queue.put(None)
        except Exception:
            pass
