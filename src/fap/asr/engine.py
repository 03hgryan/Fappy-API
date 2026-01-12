# asr.py
"""
Streaming ASR Engine using faster-whisper

Step 1: Enable word_timestamps to get per-word timing information.
"""

import time
from .buffer import RollingAudioBuffer
from .utils import is_silence


class StreamingASR:
    """
    Streaming ASR using faster-whisper with rolling buffer.
    Emits progressive hypotheses with revisions.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", model=None):
        """
        Initialize streaming ASR.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
            model: Pre-loaded WhisperModel instance (optional, reuses shared model)
        """
        print(f"🤖 Initializing ASR: {model_size} on {device}")

        # Use provided model or load a new one
        if model is not None:
            self.model = model
            print("✅ Using pre-loaded ASR model")
        else:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type="int8")
            print("✅ ASR model loaded")

        self.buffer = RollingAudioBuffer(max_duration_ms=2000)
        self.segment_id = f"seg-{int(time.time() * 1000)}"
        self.revision = 0
        self.last_text = ""

        # Silence detection
        self.silence_chunks = 0
        self.SILENCE_THRESHOLD_CHUNKS = 3  # ~1 second of silence

    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio chunk and return hypothesis if available.

        Args:
            audio: PCM16 audio bytes (16kHz)

        Returns:
            Hypothesis dict if transcription available, None otherwise
        """
        # Detect silence for segment boundaries
        if is_silence(audio):
            self.silence_chunks += 1

            # After sustained silence, start new segment
            if self.silence_chunks >= self.SILENCE_THRESHOLD_CHUNKS:
                if self.revision > 0:  # Only reset if we had content
                    self.segment_id = f"seg-{int(time.time() * 1000)}"
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

        # Run real ASR inference with word_timestamps ENABLED
        audio_float32 = self.buffer.get_float32()
        segments, info = self.model.transcribe(
            audio_float32,
            language="en",
            beam_size=1,  # Fast inference
            vad_filter=False,  # We handle silence ourselves
            word_timestamps=True,  # ENABLED: Get per-word timing!
        )

        # Extract text and word-level timestamps
        full_text = ""
        words_with_timestamps = []
        
        for segment in segments:
            full_text += segment.text.strip() + " "
            
            # Extract word timestamps if available
            if segment.words:
                for word in segment.words:
                    words_with_timestamps.append({
                        "word": word.word.strip(),
                        "start": word.start,  # seconds (relative to buffer start)
                        "end": word.end,      # seconds (relative to buffer start)
                        "probability": word.probability,
                    })
        
        full_text = full_text.strip()

        if not full_text:
            return None

        # Increment revision
        self.revision += 1

        # Convert word timestamps to absolute milliseconds
        buffer_start_ms = self.buffer.start_time_ms
        words_absolute = []
        for w in words_with_timestamps:
            words_absolute.append({
                "word": w["word"],
                "start_ms": buffer_start_ms + int(w["start"] * 1000),
                "end_ms": buffer_start_ms + int(w["end"] * 1000),
                "probability": w["probability"],
            })

        # === STEP 1 LOGGING: Show word timestamps ===
        print(f"📝 Words with timestamps (revision {self.revision}):")
        for w in words_absolute:
            print(f"   [{w['start_ms']:5d} - {w['end_ms']:5d}ms] \"{w['word']}\" (prob: {w['probability']:.2f})")

        # Build hypothesis
        hypothesis = {
            "type": "hypothesis",
            "segment_id": self.segment_id,
            "revision": self.revision,
            "text": full_text,
            "words": words_absolute,  # NEW: Word-level timestamps
            "confidence": 0.7,
            "start_time_ms": self.buffer.start_time_ms,
            "end_time_ms": self.buffer.start_time_ms + 2000,
        }

        self.last_text = full_text
        return hypothesis