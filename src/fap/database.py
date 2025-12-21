from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if os.getenv("ENVIRONMENT") == "development":
    print("✅ Database connection configured")
else:
    print("✅ Production database connection configured")

engine = create_engine(DATABASE_URL)

# Test database connection
try:
    with engine.connect() as connection:
        print("✅ Database connection successful!")
except Exception as e:
    print(f"❌ Database connection failed: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
