from fastapi import FastAPI
from app.database import engine, Base
from app.routers import books

# Auto-create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Library Management System",
    description="CRUD API for managing books using FastAPI & PostgreSQL",
    version="1.0.0"
)

app.include_router(books.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Library Management System 📚"}