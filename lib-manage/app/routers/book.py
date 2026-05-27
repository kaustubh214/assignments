from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/books", tags=["Books"])


# ── CREATE ──────────────────────────────────────────────
@router.post("/", response_model=schemas.BookResponse, status_code=status.HTTP_201_CREATED)
def create_book(book: schemas.BookCreate, db: Session = Depends(get_db)):
    new_book = models.Book(**book.model_dump())
    db.add(new_book)
    db.commit()
    db.refresh(new_book)
    return new_book


# ── READ ALL ─────────────────────────────────────────────
@router.get("/", response_model=list[schemas.BookResponse])
def get_all_books(db: Session = Depends(get_db)):
    return db.query(models.Book).all()


# ── READ ONE ─────────────────────────────────────────────
@router.get("/{book_id}", response_model=schemas.BookResponse)
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
    return book


# ── UPDATE ───────────────────────────────────────────────
@router.patch("/{book_id}", response_model=schemas.BookResponse)
def update_book(book_id: int, updates: schemas.BookUpdate, db: Session = Depends(get_db)):
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
    
    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(book, field, value)
    
    db.commit()
    db.refresh(book)
    return book


# ── DELETE ───────────────────────────────────────────────
@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_book(book_id: int, db: Session = Depends(get_db)):
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
    
    db.delete(book)
    db.commit()