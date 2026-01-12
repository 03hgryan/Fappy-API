# asr/whisper_engine.py
"""
Faster-Whisper ASR Engine implementation.

Local, offline speech recognition using faster-whisper.
"""

import time
import platform
from typing import Any

from .base import ASREngine, Hypothesis, WordInfo


class WhisperASR(ASREngine):
    """
    Streaming ASR using faster-whisper with rolling buffer.
    Emits progressive hypotheses with revisions.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        model: Any = None,
        buffer_duration_ms: int = 2000,
    ):
        """
        Initialize Whisper ASR engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (auto, cpu, cuda)
            model: Pre-loaded WhisperModel instance (optional, for shared model)
            buffer_duration_ms: Rolling buffer duration in milliseconds
        """
        print(f"🤖 Initializing Whisper ASR: {model_size} on {device}")

        # Import here to avoid import errors if not using whisper
        from .buffer import RollingAudioBuffer
        from .utils import is_silence
        
        self._is_silence = is_silence

        # Use provided model or load a new one
        if model is not None:
            self.model = model
            print("✅ Using pre-loaded Whisper model")
        else:
            self.model = self._load_model(model_size, device)

        self.buffer = RollingAudioBuffer(max_duration_ms=buffer_duration_ms)
        self.segment_id = f"whisper-seg-{int(time.time() * 1000)}"
        self.revision = 0
        self.last_text = ""

        # Silence detection
        self.silence_chunks = 0
        self.SILENCE_THRESHOLD_CHUNKS = 3  # ~1 second of silence

    def _load_model(self, model_size: str, device: str):
        """Load the Whisper model with optimal settings."""
        from faster_whisper import WhisperModel

        # Auto-detect device and compute type
        if device == "auto":
            is_apple_silicon = (
                platform.system() == "Darwin" and 
                platform.processor() == "arm"
            )

            if is_apple_silicon:
                device = "cpu"
                compute_type = "int8"
                cpu_threads = 8
                print("🍎 Apple Silicon detected - using optimized CPU inference")
            else:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                        compute_type = "float16"
                        cpu_threads = 4
                        print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
                    else:
                        device = "cpu"
                        compute_type = "int8"
                        cpu_threads = 4
                        print("💻 Using CPU")
                except ImportError:
                    device = "cpu"
                    compute_type = "int8"
                    cpu_threads = 4
                    print("💻 Using CPU")
        elif device == "cuda":
            compute_type = "float16"
            cpu_threads = 4
        else:
            compute_type = "int8"
            cpu_threads = 8

        print(f"📦 Loading {model_size} model with {compute_type} precision...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        print("✅ Whisper model loaded")
        return model

    @property
    def provider_name(self) -> str:
        return "whisper"

    def feed(self, audio: bytes) -> Hypothesis | None:
        """
        Feed audio chunk and return hypothesis if available.

        Args:
            audio: PCM16 audio bytes (16kHz)

        Returns:
            Hypothesis dict if transcription available, None otherwise
        """
        # Detect silence for segment boundaries
        if self._is_silence(audio):
            self.silence_chunks += 1

            if self.silence_chunks >= self.SILENCE_THRESHOLD_CHUNKS:
                if self.revision > 0:
                    self.segment_id = f"whisper-seg-{int(time.time() * 1000)}"
                    self.revision = 0
                    self.last_text = ""
                return None
        else:
            self.silence_chunks = 0

        # Add to buffer
        self.buffer.push(audio, chunk_duration_ms=320)

        # Wait until buffer is ready
        if not self.buffer.is_ready(min_fill_ratio=0.75):
            return None

        # Run ASR inference with word timestamps
        audio_float32 = self.buffer.get_float32()
        segments, info = self.model.transcribe(
            audio_float32,
            language="en",
            beam_size=1,
            vad_filter=False,
            word_timestamps=True,
        )

        # Extract text and word-level timestamps
        full_text = ""
        words_with_timestamps: list[WordInfo] = []

        for segment in segments:
            full_text += segment.text.strip() + " "

            if segment.words:
                for word in segment.words:
                    words_with_timestamps.append({
                        "word": word.word.strip(),
                        "start_ms": self.buffer.start_time_ms + int(word.start * 1000),
                        "end_ms": self.buffer.start_time_ms + int(word.end * 1000),
                        "probability": word.probability,
                    })

        full_text = full_text.strip()

        if not full_text:
            return None

        self.revision += 1

        # Log word timestamps
        print(f"📝 Words with timestamps (revision {self.revision}):")
        for w in words_with_timestamps:
            print(f"   [{w['start_ms']:5d} - {w['end_ms']:5d}ms] \"{w['word']}\" (prob: {w['probability']:.2f})")

        hypothesis: Hypothesis = {
            "type": "hypothesis",
            "segment_id": self.segment_id,
            "revision": self.revision,
            "text": full_text,
            "words": words_with_timestamps,
            "confidence": 0.7,
            "start_time_ms": self.buffer.start_time_ms,
            "end_time_ms": self.buffer.start_time_ms + 2000,
        }

        self.last_text = full_text
        return hypothesis

    def reset(self) -> None:
        """Reset the engine state for a new segment."""
        self.segment_id = f"whisper-seg-{int(time.time() * 1000)}"
        self.revision = 0
        self.last_text = ""
        self.silence_chunks = 0
        self.buffer.clear()