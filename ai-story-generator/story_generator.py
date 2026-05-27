# story_generator.py
from transformers import pipeline

# Load the text-generation pipeline with GPT-2
print("Loading GPT-2 model...")
generator = pipeline("text-generation", model="gpt2")
print("Model loaded.\n")

# ─── Prompts ───────────────────────────────────────────────
prompts = {
    "Space Story":          "In the year 2050, robots started",
    "Horror Story":         "The old house at the end of the street had been empty for years, until one night",
    "Motivational Paragraph": "Every great achievement begins with a single step. Today is the day",
}

# ─── Generate ──────────────────────────────────────────────
for genre, prompt in prompts.items():
    print(f"{'='*60}")
    print(f"  {genre.upper()}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}\n")

    results = generator(
        prompt,
        max_length=200,           # total tokens in output
        num_return_sequences=1,   # how many stories to generate
        temperature=0.9,          # creativity: higher = more random
        do_sample=True,           # required for temperature to work
        truncation=True,
    )

    for i, result in enumerate(results, 1):
        print(f"[Story {i}]\n{result['generated_text']}\n")