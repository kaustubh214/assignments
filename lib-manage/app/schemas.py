from pydantic import BaseModel
from typing import Optional

class BookBase(BaseModel):
    title: str
    author: str
    genre: Optional[str] = None
    year: Optional[int] = None
    price: Optional[float] = None
    available: Optional[bool] = True

class BookCreate(BookBase):
    pass

class BookUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    price: Optional[float] = None
    available: Optional[bool] = None

class BookResponse(BookBase):
    id: int

    class Config:
        from_attributes = True