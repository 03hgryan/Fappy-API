"""
Helper functions for ASR processing
"""

import struct
import math


def is_silence(pcm16_bytes: bytes, threshold: int = 300) -> bool:
    """
    Detect silence using RMS energy.

    Args:
        pcm16_bytes: PCM16 audio data
        threshold: RMS threshold (lower = more sensitive to silence)

    Returns:
        True if audio is below threshold (silence), False otherwise
    """
    if len(pcm16_bytes) < 2:
        return True

    # Unpack PCM16 samples
    num_samples = len(pcm16_bytes) // 2
    samples = struct.unpack("<" + "h" * num_samples, pcm16_bytes)

    # Calculate RMS (root mean square)
    rms = math.sqrt(sum(s * s for s in samples) / num_samples)

    return rms < threshold
