# 📚 Library Management System

A backend REST API built with **FastAPI** and **PostgreSQL**.

## Tech Stack
- FastAPI
- PostgreSQL
- SQLAlchemy ORM
- Pydantic v2

## Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file:
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/library_db
5. Run the server: `uvicorn app.main:app --reload`

## API Endpoints

| Method | Endpoint        | Description        |
|--------|-----------------|--------------------|
| POST   | /books/         | Add a new book     |
| GET    | /books/         | Get all books      |
| GET    | /books/{id}     | Get a single book  |
| PATCH  | /books/{id}     | Update a book      |
| DELETE | /books/{id}     | Delete a book      |

## Docs
Visit `http://127.0.0.1:8000/docs` for Swagger UI.