from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from fap import crud, schemas
from fap.database import get_db

router = APIRouter()

@router.post("/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)

@router.get("/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    print(f"🌐 API: GET /users/ called (skip={skip}, limit={limit})")
    users = crud.get_users(db=db, skip=skip, limit=limit)
    print(f"🔄 API: Returning {len(users)} users")
    return users

@router.get("/{user_id}/", response_model=schemas.User)
def read_user(user_id: str, db: Session = Depends(get_db)):
    db_user = crud.get_user(db=db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.put("/{user_id}/", response_model=schemas.User)
def update_user(user_id: str, user: schemas.UserUpdate, db: Session = Depends(get_db)):
    db_user = crud.update_user(db=db, user_id=user_id, user=user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.delete("/{user_id}/")
def delete_user(user_id: str, db: Session = Depends(get_db)):
    success = crud.delete_user(db=db, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}