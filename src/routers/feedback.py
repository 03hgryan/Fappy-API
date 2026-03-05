"""
Feedback router.
Endpoint: POST /feedback
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth.supabase_client import insert_feedback

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
        insert_feedback(
            message=body.message.strip(),
            email=body.email.strip() if body.email else None,
            user_id=body.user_id,
        )
        return {"success": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
