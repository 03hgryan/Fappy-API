"""
Speech-to-Text routers package.
"""

from fastapi import APIRouter
from .speechmatics import router as speechmatics_router
from .elevenlabs import router as elevenlabs_router
from .assemblyai import router as assemblyai_router
from .deepgram import router as deepgram_router

router = APIRouter()

# Mount provider routers
router.include_router(speechmatics_router, prefix="/speechmatics", tags=["speechmatics"])
router.include_router(elevenlabs_router, prefix="/elevenlabs", tags=["elevenlabs"])
router.include_router(assemblyai_router, prefix="/assemblyai", tags=["assemblyai"])
router.include_router(deepgram_router, prefix="/deepgram", tags=["deepgram"])