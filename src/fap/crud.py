from sqlalchemy.orm import Session
from fap import models, schemas
import uuid


def get_user(db: Session, user_id: str):
    print(f"🔍 Looking for user with ID: {user_id}")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    print(f"📝 Found user: {user.name if user else 'None'}")
    return user


def get_users(db: Session, skip: int = 0, limit: int = 100):
    print(f"📋 Getting users (skip={skip}, limit={limit})")
    users = db.query(models.User).offset(skip).limit(limit).all()
    print(f"📊 Found {len(users)} users")
    for user in users:
        print(f"   - {user.id}: {user.name}")
    return users


def create_user(db: Session, user: schemas.UserCreate):
    user_id = str(uuid.uuid4())
    print(f"➕ Creating user: {user.name} with ID: {user_id}")
    db_user = models.User(id=user_id, name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    print(f"✅ User created successfully: {db_user.id}")
    return db_user


def update_user(db: Session, user_id: str, user: schemas.UserUpdate):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        if user.name is not None:
            db_user.name = user.name
        db.commit()
        db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: str):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False
