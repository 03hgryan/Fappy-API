from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


def generate_cuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_cuid)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AutomationRequest(Base):
    __tablename__ = "automation_requests"

    id = Column(String, primary_key=True, default=generate_cuid)
    user_request = Column(Text, nullable=False)
    request_type = Column(
        String, nullable=False
    )  # 'analyze' or 'plan' or 'enhanced_plan'
    additional_info = Column(JSON, nullable=True)

    # Separate columns for different response types
    automation_steps = Column(JSON, nullable=True)  # For plan/enhanced_plan responses
    missing_information = Column(JSON, nullable=True)  # For analyze responses
    suggested_websites = Column(JSON, nullable=True)  # For analyze responses
    needs_more_info = Column(Boolean, nullable=True)  # For analyze responses

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
