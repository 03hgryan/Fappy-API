"""
Speech-to-Text routers package.
"""

from fastapi import APIRouter
# from .elevenlabsTwo import router as elevenlabs_router
from .elevenlabsTest import router as elevenlabs_router

router = APIRouter()

# Mount provider routers
router.include_router(elevenlabs_router, prefix="/elevenlabs", tags=["elevenlabs"])

# Add more providers here:
# from .deepgram import router as deepgram_router
# router.include_router(deepgram_router, prefix="/deepgram", tags=["deepgram"])