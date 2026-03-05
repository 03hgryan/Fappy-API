"""
API interest router.
Endpoint: POST /interest
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from database import get_engine

router = APIRouter()


class InterestRequest(BaseModel):
    email: str
    name: str | None = None
    company: str | None = None
    api_needed: str | None = None
    use_case: str | None = None


@router.post("")
async def register_interest(body: InterestRequest):
    if not body.email.strip():
        return JSONResponse({"error": "Email is required"}, status_code=400)
    try:
        with get_engine().connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO api_interest (email, name, company, api_needed, use_case, created_at) "
                    "VALUES (:email, :name, :company, :api_needed, :use_case, :created_at)"
                ),
                {
                    "email": body.email.strip().lower(),
                    "name": body.name.strip() if body.name else None,
                    "company": body.company.strip() if body.company else None,
                    "api_needed": body.api_needed.strip() if body.api_needed else None,
                    "use_case": body.use_case.strip() if body.use_case else None,
                    "created_at": datetime.now(timezone.utc),
                },
            )
            conn.commit()
        return {"success": True}
    except Exception as e:
        msg = str(e)
        if "duplicate" in msg.lower() or "unique" in msg.lower():
            return {"success": True, "already_registered": True}
        return JSONResponse({"error": msg}, status_code=500)
