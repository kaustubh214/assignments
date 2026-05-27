import gradio as gr
from transformers import pipeline

# Load model once at startup
sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text."
    
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]
    
    emoji = "😊" if label == "POSITIVE" else "😞"
    return f"{emoji} **{label}** (confidence: {score:.2%})"

# Gradio UI
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Enter text",
        placeholder="Type something like: 'I love this course!'",
        lines=3
    ),
    outputs=gr.Markdown(label="Result"),
    title="🔍 Sentiment Analyzer",
    description="Powered by DistilBERT fine-tuned on SST-2. Enter any text to detect its sentiment.",
    examples=[
        ["This project is absolutely amazing!"],
        ["I'm really disappointed with the results."],
        ["The weather is okay, nothing special."],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()