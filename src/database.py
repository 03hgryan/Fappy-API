import os
from urllib.parse import urlparse
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        raw = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
        if not raw:
            raise RuntimeError("DATABASE_URL or DIRECT_URL environment variable is required")
        p = urlparse(raw)
        url = URL.create(
            drivername=p.scheme,
            username=p.username,
            password=p.password,
            host=p.hostname,
            port=p.port,
            database=p.path.lstrip("/"),
        )
        _engine = create_engine(url)
    return _engine
