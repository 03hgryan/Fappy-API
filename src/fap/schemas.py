from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List, Dict
import re


class UserBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="User name")


class UserCreate(UserBase):
    @validator("name")
    def validate_name(cls, v):
        # Sanitize: strip whitespace
        v = v.strip()

        # Validate: only alphanumeric, spaces, hyphens, underscores
        if not re.match(r"^[a-zA-Z0-9 _-]+$", v):
            raise ValueError("Name contains invalid characters")

        # Prevent empty after strip
        if not v:
            raise ValueError("Name cannot be empty")

        return v


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)

    @validator("name")
    def validate_name(cls, v):
        if v is not None:
            v = v.strip()
            if not re.match(r"^[a-zA-Z0-9 _-]+$", v):
                raise ValueError("Name contains invalid characters")
            if not v:
                raise ValueError("Name cannot be empty")
        return v


class User(UserBase):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
