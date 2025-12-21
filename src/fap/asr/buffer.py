"""
Audio buffering for streaming ASR
"""

import struct
import numpy as np


# Audio constants
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # PCM16


class RollingAudioBuffer:
    """
    Maintains a sliding window of audio for streaming inference.

    Accumulates audio chunks and maintains a fixed-size window,
    discarding old audio when the buffer is full.
    """

    def __init__(self, max_duration_ms: int = 2000):
        """
        Initialize rolling buffer.

        Args:
            max_duration_ms: Maximum buffer duration in milliseconds
        """
        self.max_bytes = int(SAMPLE_RATE * BYTES_PER_SAMPLE * (max_duration_ms / 1000))
        self.buffer = bytearray()
        self.start_time_ms = 0

    def push(self, audio: bytes, chunk_duration_ms: int):
        """
        Add audio chunk to buffer and maintain sliding window.

        Args:
            audio: PCM16 audio bytes
            chunk_duration_ms: Duration of this chunk in milliseconds
        """
        self.buffer.extend(audio)

        # Remove old audio if buffer exceeds max size
        overflow = len(self.buffer) - self.max_bytes
        if overflow > 0:
            self.buffer = self.buffer[overflow:]
            self.start_time_ms += chunk_duration_ms

    def get_bytes(self) -> bytes:
        """Get current buffer as bytes"""
        return bytes(self.buffer)

    def get_float32(self):
        """
        Convert PCM16 buffer to float32 numpy array for Whisper.

        Returns:
            Numpy array of float32 samples normalized to [-1.0, 1.0]
        """
        if len(self.buffer) < 2:
            return np.array([], dtype=np.float32)

        # Unpack PCM16 samples
        num_samples = len(self.buffer) // 2
        pcm16 = struct.unpack("<" + "h" * num_samples, self.buffer)

        # Normalize to [-1.0, 1.0] and convert to numpy array
        return np.array([sample / 32768.0 for sample in pcm16], dtype=np.float32)

    def is_ready(self, min_fill_ratio: float = 0.75) -> bool:
        """
        Check if buffer has enough data for inference.

        Args:
            min_fill_ratio: Minimum buffer fill ratio (0.0 to 1.0)

        Returns:
            True if buffer is sufficiently filled
        """
        return len(self.buffer) >= self.max_bytes * min_fill_ratio
