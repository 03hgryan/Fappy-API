from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from src.routers import websocket
from src.routers import stt
# from src.utils.embeddings import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload LaBSE at startup
    # load_model()
    yield


app = FastAPI(
    title="ELE - WebSocket Server",
    description="FastAPI WebSocket Server",
    version="1.0.0",
    lifespan=lifespan,  # Include lifespan here
)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ELE WebSocket Server",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "websocket": "/ws/stream",
            "elevenlabs": "/stt/elevenlabs",
        },
    }


# Include routers
app.include_router(
    websocket.router,
    prefix="/ws",
    tags=["websocket"],
)

# Mount STT routers
app.include_router(
    stt.router, 
    prefix="/stt",
    tags=["stt"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ELE WebSocket Server",
    }