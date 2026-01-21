"""
Helper functions for ASR processing
"""

import struct
import math
import array


def resample_audio(audio_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Resample PCM16 audio between sample rates using linear interpolation.

    Args:
        audio_bytes: PCM16 audio data (little-endian, mono)
        from_rate: Source sample rate in Hz (e.g., 16000)
        to_rate: Target sample rate in Hz (e.g., 24000)

    Returns:
        Resampled PCM16 audio bytes

    Example:
        # Resample from 16kHz to 24kHz for OpenAI
        resampled = resample_audio(audio_16k, 16000, 24000)
    """
    if from_rate == to_rate:
        return audio_bytes

    if len(audio_bytes) < 2:
        return audio_bytes

    # Unpack input samples
    num_samples = len(audio_bytes) // 2
    samples = struct.unpack("<" + "h" * num_samples, audio_bytes)

    # Calculate output size
    ratio = to_rate / from_rate
    out_samples = int(num_samples * ratio)

    if out_samples == 0:
        return audio_bytes

    # Linear interpolation resampling
    output = array.array("h")

    for i in range(out_samples):
        # Position in source
        src_pos = i / ratio
        src_idx = int(src_pos)
        frac = src_pos - src_idx

        # Get samples for interpolation
        if src_idx >= num_samples - 1:
            sample = samples[-1]
        else:
            s0 = samples[src_idx]
            s1 = samples[src_idx + 1]
            sample = int(s0 + frac * (s1 - s0))

        # Clamp to int16 range
        sample = max(-32768, min(32767, sample))
        output.append(sample)

    return output.tobytes()


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
