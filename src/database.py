import os
from sqlalchemy import create_engine

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL or DIRECT_URL environment variable is required")
        _engine = create_engine(url)
    return _engine
