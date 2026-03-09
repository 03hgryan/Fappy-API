"""
Google OAuth2 authentication router.
Endpoints: /auth/google/login, /auth/google/callback, /auth/me
"""

import secrets
from urllib.parse import urlencode, urlparse, parse_qs

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy import text

from auth.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, AUTH_ENABLED
from auth.jwt_utils import create_access_token
from auth.dependencies import get_current_user
from database import get_engine

router = APIRouter()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/google/login")
async def google_login(
    redirect_uri: str = Query(default=""),
    callback_url: str = Query(default=""),
):
    if not AUTH_ENABLED:
        return JSONResponse({"error": "Auth is disabled"}, status_code=400)

    # callback_url is where Google sends the code (our /auth/google/callback)
    # redirect_uri is where we redirect after minting the JWT (extension URL)
    if not callback_url:
        callback_url = "http://localhost:8000/auth/google/callback"

    state = secrets.token_urlsafe(16)
    if redirect_uri:
        state += "|" + redirect_uri

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": callback_url,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "state": state,
        "prompt": "consent",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(default=""),
):
    if not AUTH_ENABLED:
        return JSONResponse({"error": "Auth is disabled"}, status_code=400)

    # Parse extension redirect URI from state
    ext_redirect = ""
    if "|" in state:
        _, ext_redirect = state.split("|", 1)

    # Determine callback URL (must exactly match what was used in /login)
    callback_url = str(request.url).split("?")[0]

    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": callback_url,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            return JSONResponse(
                {"error": "Token exchange failed", "detail": token_resp.text},
                status_code=400,
            )
        tokens = token_resp.json()

        # Get user info
        userinfo_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        if userinfo_resp.status_code != 200:
            return JSONResponse(
                {"error": "Failed to get user info"},
                status_code=400,
            )
        userinfo = userinfo_resp.json()

    # Upsert user in database
    with get_engine().connect() as conn:
        result = conn.execute(
            text(
                "INSERT INTO users (google_id, email, name, picture_url, last_login) "
                "VALUES (:google_id, :email, :name, :picture_url, :last_login) "
                "ON CONFLICT (google_id) DO UPDATE SET "
                "  email = EXCLUDED.email, "
                "  name = EXCLUDED.name, "
                "  picture_url = EXCLUDED.picture_url, "
                "  last_login = EXCLUDED.last_login, "
                "  updated_at = now() "
                "RETURNING id"
            ),
            {
                "google_id": userinfo["id"],
                "email": userinfo["email"],
                "name": userinfo.get("name"),
                "picture_url": userinfo.get("picture"),
                "last_login": datetime.now(timezone.utc),
            },
        )
        conn.commit()
        row = result.fetchone()
    user = {"id": row[0] if row else userinfo["id"]}

    # Mint JWT
    jwt_token = create_access_token(
        user_id=str(user.get("id", userinfo["id"])),
        email=userinfo["email"],
    )

    # If extension redirect URI is set, redirect there with token
    if ext_redirect:
        separator = "&" if "?" in ext_redirect else "?"
        return RedirectResponse(
            f"{ext_redirect}{separator}token={jwt_token}"
            f"&name={userinfo.get('name', '')}"
            f"&email={userinfo['email']}"
            f"&picture={userinfo.get('picture', '')}",
        )

    # Otherwise return JSON
    return JSONResponse({
        "token": jwt_token,
        "user": {
            "email": userinfo["email"],
            "name": userinfo.get("name"),
            "picture": userinfo.get("picture"),
        },
    })


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {"user": user}
