from sqlalchemy import Column, Integer, String, Float, Boolean
from app.database import Base

class Book(Base):
    __tablename__ = "books"

    id       = Column(Integer, primary_key=True, index=True)
    title    = Column(String, nullable=False)
    author   = Column(String, nullable=False)
    genre    = Column(String, nullable=True)
    year     = Column(Integer, nullable=True)
    price    = Column(Float, nullable=True)
    available = Column(Boolean, default=True)