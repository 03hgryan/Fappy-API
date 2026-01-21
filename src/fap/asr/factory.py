# asr/factory.py
"""
Factory function for creating ASR engines.

Usage:
    from fap.asr import create_asr_engine
    
    # Local Whisper (default)
    asr = create_asr_engine("whisper", model_size="medium")
    
    # Google Cloud
    asr = create_asr_engine("google", language_code="en-US")
    
    # From environment variable
    asr = create_asr_engine()  # Uses ASR_PROVIDER env var
"""

import os
from typing import Any

from .base import ASREngine


def create_asr_engine(
    provider: str | None = None,
    **kwargs: Any,
) -> ASREngine:
    """
    Create an ASR engine instance.

    Args:
        provider: ASR provider ("whisper", "google", or None for env var)
        **kwargs: Provider-specific arguments
        
    Provider-specific kwargs:
        whisper:
            - model_size: str = "medium"
            - device: str = "auto"
            - model: WhisperModel = None (pre-loaded model)
            - buffer_duration_ms: int = 2000
            
        google:
            - language_code: str = "en-US"
            - sample_rate_hz: int = 16000
            - credentials_path: str = None
            - interim_results: bool = True

    Returns:
        ASREngine instance

    Example:
        # Whisper with custom model size
        asr = create_asr_engine("whisper", model_size="large")
        
        # Google Cloud with Korean
        asr = create_asr_engine("google", language_code="ko-KR")
    """
    # Get provider from env if not specified
    if provider is None:
        provider = os.getenv("ASR_PROVIDER", "whisper").lower()

    print(f"🔧 Creating ASR engine: {provider}")

    if provider == "whisper":
        from .whisper_engine import WhisperASR
        
        # Default kwargs for whisper
        whisper_defaults = {
            "model_size": os.getenv("WHISPER_MODEL_SIZE", "medium"),
            "device": os.getenv("WHISPER_DEVICE", "auto"),
        }
        whisper_defaults.update(kwargs)
        
        return WhisperASR(**whisper_defaults)

    elif provider == "google":
        from .google_engine import GoogleCloudASR

        # Default kwargs for google
        google_defaults = {
            "language_code": os.getenv("GOOGLE_ASR_LANGUAGE", "en-US"),
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        }
        google_defaults.update(kwargs)

        return GoogleCloudASR(**google_defaults)

    elif provider == "openai":
        from .openai_engine import OpenAIRealtimeASR

        # Default kwargs for openai
        openai_defaults = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_ASR_MODEL", "gpt-4o-transcribe"),
            "language": os.getenv("OPENAI_ASR_LANGUAGE", "en"),
        }
        openai_defaults.update(kwargs)

        return OpenAIRealtimeASR(**openai_defaults)

    else:
        raise ValueError(
            f"Unknown ASR provider: {provider}. "
            f"Supported providers: whisper, google, openai"
        )


def load_shared_model(provider: str = "whisper", **kwargs: Any) -> Any:
    """
    Load a shared model for use across multiple ASR instances.
    
    This is useful for FastAPI startup to load the model once.
    
    Args:
        provider: ASR provider
        **kwargs: Provider-specific arguments
        
    Returns:
        Loaded model (WhisperModel for whisper, None for google)
        
    Example:
        # In main.py startup
        app.state.asr_model = load_shared_model("whisper", model_size="medium")
        
        # In websocket handler
        asr = create_asr_engine("whisper", model=app.state.asr_model)
    """
    if provider == "whisper":
        import platform
        from faster_whisper import WhisperModel
        
        model_size = kwargs.get("model_size", os.getenv("WHISPER_MODEL_SIZE", "medium"))
        device = kwargs.get("device", os.getenv("WHISPER_DEVICE", "auto"))
        
        # Auto-detect settings
        if device == "auto":
            is_apple_silicon = (
                platform.system() == "Darwin" and 
                platform.processor() == "arm"
            )
            
            if is_apple_silicon:
                device = "cpu"
                compute_type = "int8"
                cpu_threads = 8
                print("🍎 Apple Silicon detected")
            else:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                        compute_type = "float16"
                        cpu_threads = 4
                        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        device = "cpu"
                        compute_type = "int8"
                        cpu_threads = 4
                except ImportError:
                    device = "cpu"
                    compute_type = "int8"
                    cpu_threads = 4
        elif device == "cuda":
            compute_type = "float16"
            cpu_threads = 4
        else:
            compute_type = "int8"
            cpu_threads = 8
        
        print(f"📦 Loading {model_size} Whisper model...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        print("✅ Whisper model loaded")
        return model
        
    elif provider == "google":
        # Google Cloud doesn't need a pre-loaded model
        print("✅ Google Cloud ASR (no pre-loading needed)")
        return None

    elif provider == "openai":
        # OpenAI Realtime doesn't need a pre-loaded model
        print("✅ OpenAI Realtime ASR (no pre-loading needed)")
        return None

    else:
        raise ValueError(f"Unknown provider: {provider}")