"""
Feedback router.
Endpoint: POST /feedback
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from database import get_engine

router = APIRouter()


class FeedbackRequest(BaseModel):
    message: str
    email: str | None = None
    user_id: str | None = None


@router.post("")
async def submit_feedback(body: FeedbackRequest):
    if not body.message.strip():
        return JSONResponse({"error": "Message is required"}, status_code=400)
    try:
        with get_engine().connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO feedback (message, email, user_id, created_at) "
                    "VALUES (:message, :email, :user_id, :created_at)"
                ),
                {
                    "message": body.message.strip(),
                    "email": body.email.strip() if body.email else None,
                    "user_id": body.user_id,
                    "created_at": datetime.now(timezone.utc),
                },
            )
            conn.commit()
        return {"success": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
