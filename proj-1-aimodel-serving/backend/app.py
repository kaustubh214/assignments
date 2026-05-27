import os
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Model Serving API")

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
origins = [origin.strip() for origin in allowed_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("📦 Loading local models...")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    gpt2_pipeline = pipeline(
        "text-generation",
        model="gpt2"
    )
    print("✅ Local models loaded!")
except Exception as e:
    print(f"❌ Error loading models: {e}")


class TextRequest(BaseModel):
    text: str


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    model: str = "zephyr"  # "zephyr" | "gpt2"


@app.get("/")
def health_check():
    return {"status": "healthy", "message": "AI Serving Engine is active"}


@app.post("/api/sentiment")
def analyze_sentiment(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be blank.")
    try:
        result = sentiment_pipeline(request.text)[0]
        return {"label": result["label"], "score": round(result["score"], 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
def generate_text(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be blank.")

    # ── GPT-2 local path ──
    if request.model == "gpt2":
        try:
            result = gpt2_pipeline(
                request.prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                truncation=True,
                num_return_sequences=1,
            )
            return {"generated_text": result[0]["generated_text"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Llama 3.3 70B via Groq ──
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment.")

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, knowledgeable assistant. Give clear and accurate responses."
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            max_tokens=request.max_tokens,
            temperature=0.7,
        )
        return {"generated_text": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)