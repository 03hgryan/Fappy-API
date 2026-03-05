from fastapi import WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException
from auth.config import AUTH_ENABLED
from auth.jwt_utils import verify_token

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict | None:
    if not AUTH_ENABLED:
        return {"sub": "dev", "email": "dev@localhost"}
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


async def require_ws_auth(ws: WebSocket) -> dict | None:
    if not AUTH_ENABLED:
        return {"sub": "dev", "email": "dev@localhost"}
    token = ws.query_params.get("token")
    if not token:
        return {"sub": "guest", "email": None}
    payload = verify_token(token)
    if payload is None:
        await ws.send_json({"type": "error", "message": "Invalid or expired token"})
        await ws.close(code=4001)
        return None
    return payload
