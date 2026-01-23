from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from fap import models
from fap.database import engine
from fap.routers import users
from fap.routers.websocket import router as websocket_router

load_dotenv()

# Only creates when tables do not exist
models.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ASR model on startup and cleanup on shutdown"""
    print("🚀 Loading ASR model on startup...")
    
    try:
        import time
        from fap.asr.factory import load_shared_model
        
        start_time = time.time()
        
        # Get provider from env (default: whisper)
        provider = os.getenv("ASR_PROVIDER", "whisper")
        print(f"📡 ASR Provider: {provider}")
        
        # Load shared model
        app.state.asr_provider = provider
        app.state.asr_model = load_shared_model(
            provider=provider,
            model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
        )
        
        load_time = time.time() - start_time
        print(f"✅ ASR ready in {load_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Failed to load ASR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback to base whisper
        print("⚠️ Attempting fallback to base Whisper model...")
        try:
            from faster_whisper import WhisperModel
            app.state.asr_provider = "whisper"
            app.state.asr_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ Fallback: base Whisper model loaded")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            app.state.asr_provider = None
            app.state.asr_model = None

    yield

    # Cleanup on shutdown
    print("🔌 Shutting down...")


app = FastAPI(
    title="AST - ASR + translation",
    description="FastAPI for AST",
    version="1.0.0",
    lifespan=lifespan,
)

# Security headers for production only
if os.getenv("ENVIRONMENT") == "production":
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            return response

    app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
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
        "message": "API for ast",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "users": "/api/users",
            "websocket": {
                "whisper": "/ws/whisper",
                "google": "/ws/google",
                "openai": "/ws/openai",
                "stream": "/ws/stream (deprecated)",
            },
        },
    }


# Include routers
app.include_router(
    users.router,
    prefix="/api/users",
    tags=["users"],
    responses={404: {"description": "User not found"}},
)

app.include_router(
    websocket_router,
    prefix="/ws",
    tags=["websocket"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "FastAPI for AST",
        "asr_provider": getattr(app.state, "asr_provider", None),
    }