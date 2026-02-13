import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from src.routers import websocket
from src.routers import stt



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload LaBSE at startup
    # load_model()
    yield


app = FastAPI(
    title="API for AST",
    description="FastAPI WebSocket Server for AST",
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
        "message": "API for AST",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "websocket": "/ws/stream",
            "elevenlabs": "/stt/elevenlabs",
            "speechmatics": "/stt/speechmatics",
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
        "service": "API for AST",
    }