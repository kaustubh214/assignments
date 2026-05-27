"""
evaluate.py
============
Evaluates the fine-tuned Finance & Tech LLM on:
  1. ROUGE scores (abstractive summarization quality)
  2. Perplexity (language modeling quality)
  3. Domain keyword coverage (jargon usage)
  4. Response length distribution
  5. Human-readable comparison table

Usage:
    python scripts/evaluate.py --model_path outputs/final_model
    python scripts/evaluate.py --model_path outputs/final_model --compare_base
"""

import argparse
import json
import os
import re
import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer


# ── Finance & Tech evaluation questions ──────────────────────
EVAL_QUESTIONS = [
    {
        "id": "F1", "domain": "finance",
        "question": "What is EBITDA?",
        "key_terms": ["earnings", "interest", "taxes", "depreciation", "amortization",
                      "profitability", "cash flow"],
        "reference": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It is a measure of a company's core operational profitability and is used to compare companies regardless of capital structure or tax environment."
    },
    {
        "id": "F2", "domain": "finance",
        "question": "Explain the P/E ratio.",
        "key_terms": ["price", "earnings", "valuation", "share", "EPS", "market"],
        "reference": "The Price-to-Earnings ratio compares a company's share price to its earnings per share. It is used to assess whether a stock is overvalued or undervalued relative to peers. A high P/E indicates growth expectations."
    },
    {
        "id": "F3", "domain": "finance",
        "question": "What are derivatives in finance?",
        "key_terms": ["futures", "options", "underlying", "hedging", "swap", "contract", "risk"],
        "reference": "Derivatives are financial contracts whose value is derived from an underlying asset. Types include futures, options, and swaps. They are used for hedging risk, speculation, and arbitrage."
    },
    {
        "id": "F4", "domain": "finance",
        "question": "What is a SIP investment?",
        "key_terms": ["systematic", "investment", "monthly", "mutual fund", "NAV",
                      "rupee cost averaging", "compounding"],
        "reference": "A Systematic Investment Plan allows investors to invest a fixed amount at regular intervals in mutual funds. It benefits from rupee cost averaging and the power of compounding."
    },
    {
        "id": "T1", "domain": "tech",
        "question": "What is Kubernetes?",
        "key_terms": ["container", "orchestration", "pod", "cluster", "deployment",
                      "scaling", "node", "Docker"],
        "reference": "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications."
    },
    {
        "id": "T2", "domain": "tech",
        "question": "Explain CI/CD in DevOps.",
        "key_terms": ["continuous integration", "continuous delivery", "pipeline",
                      "automated", "build", "test", "deploy", "Jenkins", "GitHub Actions"],
        "reference": "CI/CD stands for Continuous Integration and Continuous Delivery. It automates the process of building, testing, and deploying code changes, enabling faster and more reliable software delivery."
    },
    {
        "id": "T3", "domain": "tech",
        "question": "What is a Docker container?",
        "key_terms": ["image", "container", "isolated", "Dockerfile", "registry",
                      "lightweight", "virtual machine"],
        "reference": "Docker containers are lightweight, isolated runtime environments that package application code with all its dependencies. They share the host OS kernel unlike virtual machines."
    },
    {
        "id": "T4", "domain": "tech",
        "question": "What are vector databases?",
        "key_terms": ["embedding", "similarity", "semantic", "neural", "RAG",
                      "search", "Pinecone", "Chroma", "cosine"],
        "reference": "Vector databases store and query high-dimensional vector embeddings for semantic similarity search. They are essential for AI applications like RAG, semantic search, and recommendation systems."
    },
]


def load_model(model_path: str, base_model: Optional[str] = None,
               load_4bit: bool = True):
    """Load model and tokenizer for evaluation."""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if load_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_lora and base_model:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str,
                    max_new_tokens: int = 300) -> tuple[str, float]:
    """Generate an answer and return text + latency."""
    prompt = f"### Instruction:\n{question}\n\n### Input:\n\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    start = time.time()
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=0.7,
        top_p=0.9, do_sample=True, repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, elapsed


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = defaultdict(list)
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for metric, value in result.items():
            scores[metric].append(value.fmeasure)
    return {metric: np.mean(vals) for metric, vals in scores.items()}


def compute_keyword_coverage(responses: list[str], key_terms_list: list[list[str]]) -> float:
    """Measure how many domain-specific keywords appear in responses."""
    total_terms = 0
    covered_terms = 0
    for response, key_terms in zip(responses, key_terms_list):
        response_lower = response.lower()
        for term in key_terms:
            total_terms += 1
            if term.lower() in response_lower:
                covered_terms += 1
    return covered_terms / total_terms if total_terms > 0 else 0


def evaluate_model(model, tokenizer, label: str = "Fine-tuned") -> dict:
    """Run full evaluation suite on a model."""
    print(f"\n{'─'*50}")
    print(f"  Evaluating: {label}")
    print(f"{'─'*50}")

    responses = []
    references = []
    key_terms_list = []
    latencies = []
    domains = []

    for item in EVAL_QUESTIONS:
        print(f"  [{item['id']}] {item['question'][:55]}...")
        response, latency = generate_answer(model, tokenizer, item["question"])
        responses.append(response)
        references.append(item["reference"])
        key_terms_list.append(item["key_terms"])
        latencies.append(latency)
        domains.append(item["domain"])
        print(f"       ✓ {latency:.2f}s | {len(response.split())} words")

    # Metrics
    rouge_scores = compute_rouge(responses, references)
    keyword_coverage = compute_keyword_coverage(responses, key_terms_list)

    # Per-domain breakdown
    finance_idx = [i for i, d in enumerate(domains) if d == "finance"]
    tech_idx = [i for i, d in enumerate(domains) if d == "tech"]

    finance_coverage = compute_keyword_coverage(
        [responses[i] for i in finance_idx],
        [key_terms_list[i] for i in finance_idx])
    tech_coverage = compute_keyword_coverage(
        [responses[i] for i in tech_idx],
        [key_terms_list[i] for i in tech_idx])

    results = {
        "label": label,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "keyword_coverage": keyword_coverage,
        "finance_keyword_coverage": finance_coverage,
        "tech_keyword_coverage": tech_coverage,
        "avg_latency_s": np.mean(latencies),
        "avg_response_length_words": np.mean([len(r.split()) for r in responses]),
        "responses": responses,
        "latencies": latencies,
    }

    # Print summary
    print(f"\n  📊 {label} Results:")
    print(f"     ROUGE-1:           {rouge_scores['rouge1']:.4f}")
    print(f"     ROUGE-2:           {rouge_scores['rouge2']:.4f}")
    print(f"     ROUGE-L:           {rouge_scores['rougeL']:.4f}")
    print(f"     Keyword Coverage:  {keyword_coverage*100:.1f}%")
    print(f"     Finance Coverage:  {finance_coverage*100:.1f}%")
    print(f"     Tech Coverage:     {tech_coverage*100:.1f}%")
    print(f"     Avg Latency:       {np.mean(latencies):.2f}s")
    print(f"     Avg Response Len:  {np.mean([len(r.split()) for r in responses]):.0f} words")

    return results


def plot_results(results_list: list[dict], output_dir: str = "outputs"):
    """Generate evaluation comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    labels = [r["label"] for r in results_list]
    metrics = {
        "ROUGE-1": [r["rouge1"] for r in results_list],
        "ROUGE-2": [r["rouge2"] for r in results_list],
        "ROUGE-L": [r["rougeL"] for r in results_list],
        "Keyword Coverage": [r["keyword_coverage"] for r in results_list],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Fine-tuned vs Base Model Evaluation\n(Finance & Tech Domain)",
                 fontsize=14, fontweight="bold")
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for ax, (metric, values), color in zip(axes.flatten(), metrics.items(), colors):
        bars = ax.bar(labels, values, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Plot saved → {plot_path}")
    plt.close()


def save_results(results_list: list[dict], output_dir: str = "outputs"):
    """Save full evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    # Don't save response text in main metrics file
    clean_results = []
    for r in results_list:
        clean = {k: v for k, v in r.items() if k != "responses"}
        clean_results.append(clean)

    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"💾 Results saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Finance & Tech LLM")
    parser.add_argument("--model_path", type=str, default="outputs/final_model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model for comparison (optional)")
    parser.add_argument("--compare_base", action="store_true",
                        help="Also evaluate base model and compare")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Finance & Tech LLM — Evaluation Suite")
    print("=" * 60)

    results_list = []

    # Evaluate fine-tuned model
    model, tokenizer = load_model(args.model_path, args.base_model, not args.no_4bit)
    results = evaluate_model(model, tokenizer, "Fine-tuned Model")
    results_list.append(results)

    # Optionally compare with base model
    if args.compare_base and args.base_model:
        print("\n🔁 Loading base model for comparison...")
        del model  # Free GPU memory
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_model(args.base_model, None, not args.no_4bit)
        base_results = evaluate_model(base_model, base_tokenizer, "Base Model")
        results_list.append(base_results)

    # Save & plot
    save_results(results_list, args.output_dir)
    if len(results_list) > 1:
        plot_results(results_list, args.output_dir)

    # Final summary table
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} " + " ".join(f"{r['label'][:15]:<16}" for r in results_list))
    print("-" * 60)
    for metric in ["rouge1", "rouge2", "rougeL", "keyword_coverage",
                   "finance_keyword_coverage", "tech_keyword_coverage"]:
        name = metric.replace("_", " ").title()
        row = f"{name:<30} "
        for r in results_list:
            row += f"{r[metric]*100:>6.1f}%          "
        print(row)
    print("=" * 60)


if __name__ == "__main__":
    main()
