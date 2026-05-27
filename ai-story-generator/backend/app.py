from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import torch

app = Flask(__name__)
CORS(app)

print("Loading GPT-2...")
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1,
)
print("Ready.")

GENRE_STARTERS = {
    "space":        "In the year 2050, robots started",
    "horror":       "The old house at the end of the street had been empty for years, until one night",
    "motivational": "Every great achievement begins with a single step. Today is the day",
}

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    genre       = data.get("genre", "space").lower()
    custom_prompt = data.get("prompt", "").strip()
    max_length  = int(data.get("max_length", 200))
    temperature = float(data.get("temperature", 0.9))
    sequences   = int(data.get("num_return_sequences", 1))

    prompt = custom_prompt if custom_prompt else GENRE_STARTERS.get(genre, GENRE_STARTERS["space"])

    results = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=sequences,
        temperature=temperature,
        do_sample=True,
        truncation=True,
    )

    stories = [r["generated_text"] for r in results]
    return jsonify({"prompt": prompt, "genre": genre, "stories": stories})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)