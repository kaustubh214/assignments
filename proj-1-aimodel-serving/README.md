# AI Model Hub

AI Model Hub is a full-stack app with a FastAPI backend and a Vite React frontend. The backend serves sentiment analysis with a local Hugging Face Transformers pipeline and text generation with either GPT-2 locally or Llama 3.3 70B through the Groq API.

## Tech Stack

- Backend: FastAPI, Uvicorn, Transformers, PyTorch, Groq SDK
- Frontend: React, Vite
- Backend deployment target: Hugging Face Spaces
- Frontend deployment target: Vercel

## Project Structure

```text
.
+-- backend/
|   +-- app.py
|   +-- requirements.txt
|   +-- Dockerfile
|   +-- .env.example
+-- frontend/
|   +-- src/
|   +-- package.json
|   +-- .env.example
+-- README.md
```

## Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer
- A Groq API key for Llama generation
- Git

## Environment Variables

Create local env files from the examples:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

On Windows PowerShell:

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
```

### Backend Variables

```env
GROQ_API_KEY=your_groq_api_key_here
ALLOWED_ORIGINS=http://localhost:5173
HOST=127.0.0.1
PORT=8000
```

### Frontend Variables

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Local Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd python-train
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `backend/.env` and add your values:

```bash
cp .env.example .env
```

Run the backend:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open this URL to verify the API:

```text
http://127.0.0.1:8000/
```

### 3. Set Up the Frontend

Open a new terminal:

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```bash
cp .env.example .env
```

Run the frontend:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

## API Endpoints

### Health Check

```http
GET /
```

### Sentiment Analysis

```http
POST /api/sentiment
Content-Type: application/json

{
  "text": "This deployment works well."
}
```

### Text Generation

```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "Explain machine learning simply.",
  "max_tokens": 512,
  "model": "zephyr"
}
```

Use `"model": "gpt2"` to run local GPT-2 generation.

## Deploy Backend on Hugging Face Spaces

Hugging Face Spaces can run this FastAPI app as a Docker Space. Docker is the recommended option for this backend because the app needs an API server and model dependencies.

### 1. Create a New Space

1. Go to Hugging Face Spaces.
2. Click **Create new Space**.
3. Choose a Space name.
4. Select **Docker** as the Space SDK.
5. Choose the hardware. CPU works, but model loading can be slow. Upgrade hardware if needed.
6. Create the Space.

### 2. Add Backend Files to the Space

Upload or push these files from the `backend` folder to the root of the Hugging Face Space repository:

```text
app.py
requirements.txt
Dockerfile
```

The provided `backend/Dockerfile` contains:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 3. Configure Space Secrets

In your Space, go to **Settings > Variables and secrets** and add:

```text
GROQ_API_KEY=<your Groq API key>
ALLOWED_ORIGINS=https://your-vercel-app.vercel.app,http://localhost:5173
PORT=7860
HOST=0.0.0.0
```

Do not commit `.env` files or API keys to GitHub.

### 4. Get the Backend URL

After the Space builds, your backend URL will look like:

```text
https://<your-username>-<your-space-name>.hf.space
```

Test:

```text
https://<your-username>-<your-space-name>.hf.space/
```

You should see:

```json
{
  "status": "healthy",
  "message": "AI Serving Engine is active"
}
```

## Deploy Frontend on Vercel

### 1. Push the Project to GitHub

```bash
git add .
git commit -m "Add deployment documentation"
git push origin main
```

### 2. Import the Project in Vercel

1. Go to Vercel.
2. Click **Add New Project**.
3. Import your GitHub repository.
4. Set the root directory to:

```text
frontend
```

### 3. Configure Build Settings

Use these settings:

```text
Framework Preset: Vite
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

### 4. Add Environment Variable

In Vercel project settings, add:

```text
VITE_API_BASE_URL=https://<your-username>-<your-space-name>.hf.space
```

Redeploy the frontend after adding or changing environment variables.

### 5. Update Backend CORS

After Vercel gives you the frontend URL, add it to the Hugging Face Space `ALLOWED_ORIGINS` variable:

```text
ALLOWED_ORIGINS=https://your-vercel-app.vercel.app,http://localhost:5173
```

Restart or rebuild the Space after changing the variable.

## Production Checklist

- `backend/.env` is not committed.
- `frontend/.env` is not committed.
- `GROQ_API_KEY` is stored as a Hugging Face Space secret.
- `VITE_API_BASE_URL` in Vercel points to the Hugging Face Space URL.
- `ALLOWED_ORIGINS` in Hugging Face includes the Vercel frontend URL.
- The backend health check URL returns a healthy response.
- The Vercel frontend can call `/api/sentiment` and `/api/generate`.

## Common Issues

### Frontend Cannot Connect to Backend

Check `VITE_API_BASE_URL` in Vercel. It must be the full Hugging Face Space URL without a trailing slash.

### CORS Error

Add your Vercel URL to `ALLOWED_ORIGINS` in Hugging Face Spaces.

### Groq Generation Fails

Confirm `GROQ_API_KEY` is set in Hugging Face Space secrets. Local GPT-2 generation can still work without Groq, but the Llama option requires the key.

### Backend Takes Time to Start

The backend loads Transformer models at startup. First boot can take longer on CPU Spaces.
