import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from src.routers import websocket
from src.routers import stt
from src.routers import auth as auth_router
from src.routers import feedback as feedback_router
from src.routers import interest as interest_router
from src.auth.config import AUTH_ENABLED



@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Auth {'ENABLED' if AUTH_ENABLED else 'DISABLED'}")
    yield


app = FastAPI(
    title="API for AST",
    description="FastAPI WebSocket Server for AST",
    version="1.0.1",
    lifespan=lifespan,  # Include lifespan here
)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS").split(",")
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
        "version": "1.0.1",
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

# Mount auth router
app.include_router(
    auth_router.router,
    prefix="/auth",
    tags=["auth"],
)

# Mount feedback router
app.include_router(
    feedback_router.router,
    prefix="/feedback",
    tags=["feedback"],
)

# Mount interest router
app.include_router(
    interest_router.router,
    prefix="/interest",
    tags=["interest"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "API for AST",
    }