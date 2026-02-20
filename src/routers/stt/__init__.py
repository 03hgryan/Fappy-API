"""
Speech-to-Text routers package.
"""

from fastapi import APIRouter
from .speechmatics import router as speechmatics_router
from .elevenlabs import router as elevenlabs_router
router = APIRouter()

# Mount provider routers
router.include_router(speechmatics_router, prefix="/speechmatics", tags=["speechmatics"])
router.include_router(elevenlabs_router, prefix="/elevenlabs", tags=["elevenlabs"])