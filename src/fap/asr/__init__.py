"""
ASR module for streaming speech recognition
"""

"""
ASR Engine abstraction layer.

Supports multiple backends:
- faster-whisper (local, offline)
- Google Cloud Speech-to-Text (cloud, streaming)
- OpenAI Realtime API (cloud, streaming)

Usage:
    # Using adapter (recommended)
    from fap.asr import create_asr_adapter

    adapter = create_asr_adapter("whisper", model_size="medium")
    # or
    adapter = create_asr_adapter("google", language_code="en-US")
    # or
    adapter = create_asr_adapter("openai", model="gpt-4o-transcribe")

    hypothesis = adapter.feed(audio_bytes)

    # Using engine directly
    from fap.asr import create_asr_engine

    engine = create_asr_engine("whisper", model_size="medium")
"""

from .engine import StreamingASR
from .buffer import RollingAudioBuffer
from .utils import is_silence, resample_audio
from .base import ASREngine
from .whisper_engine import WhisperASR
from .google_engine import GoogleCloudASR
from .openai_engine import OpenAIRealtimeASR
from .factory import create_asr_engine, load_shared_model
from .adapter import ASRAdapter, WhisperAdapter, GoogleAdapter, OpenAIAdapter, create_asr_adapter


__all__ = [
    "StreamingASR",
    "RollingAudioBuffer",
    "is_silence",
    "resample_audio",
    "ASREngine",
    "WhisperASR",
    "GoogleCloudASR",
    "OpenAIRealtimeASR",
    "create_asr_engine",
    "load_shared_model",
    "ASRAdapter",
    "WhisperAdapter",
    "GoogleAdapter",
    "OpenAIAdapter",
    "create_asr_adapter",
]
