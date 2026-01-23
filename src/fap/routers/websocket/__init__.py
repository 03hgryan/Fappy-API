# routers/websocket/__init__.py
"""
WebSocket router module for ASR streaming.

Provides separate WebSocket routes per ASR provider for lower latency:
- /ws/whisper - Local Whisper ASR (incremental mode)
- /ws/google  - Google Cloud Speech-to-Text (rewriting mode)
- /ws/openai  - OpenAI Realtime API (rewriting mode)
- /ws/stream  - Legacy endpoint (deprecated, uses adapter pattern)
"""

from fastapi import APIRouter

from .whisper import router as whisper_router
from .google import router as google_router
from .openai import router as openai_router
from .legacy import router as legacy_router

# Aggregate all routers
router = APIRouter()
router.include_router(whisper_router)
router.include_router(google_router)
router.include_router(openai_router)
router.include_router(legacy_router)

__all__ = ["router"]
