"""
inference.py
=============
Load a fine-tuned Finance & Tech domain LLM and run interactive inference.
Supports both the fine-tuned LoRA model and the base model for comparison.

Usage:
    # Interactive chat with fine-tuned model
    python scripts/inference.py --model_path outputs/final_model

    # Single prompt
    python scripts/inference.py --model_path outputs/final_model --prompt "What is EBITDA?"

    # Compare fine-tuned vs base model
    python scripts/inference.py --model_path outputs/final_model --compare

    # Run all demo questions
    python scripts/inference.py --model_path outputs/final_model --demo
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ── Prompt template (must match training) ───────────────────
PROMPT_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:"""

# ── Demo Questions ────────────────────────────────────────────
DEMO_QUESTIONS = [
    # Finance
    ("What is EBITDA and how is it calculated?", "finance"),
    ("Explain the P/E ratio and give an example.", "finance"),
    ("What are financial derivatives? Name 3 types.", "finance"),
    ("What is equity dilution and how can shareholders protect themselves?", "finance"),
    ("Explain SIP investments and the benefit of rupee cost averaging.", "finance"),
    ("What are the key steps in balance sheet analysis?", "finance"),
    # Tech
    ("What is Kubernetes and what problems does it solve?", "tech"),
    ("Explain microservices vs monolithic architecture.", "tech"),
    ("What is CI/CD? Describe a typical pipeline.", "tech"),
    ("What are Docker containers? How do they differ from VMs?", "tech"),
    ("What are vector databases and why do AI apps need them?", "tech"),
    ("What is a service mesh in microservices? Give an example.", "tech"),
    # Cross-domain
    ("How would you use a vector database to build a semantic search for financial reports?", "cross-domain"),
    ("Explain how CI/CD pipelines can be applied to machine learning model deployment (MLOps).", "cross-domain"),
]


class DomainLLM:
    """Wrapper for the fine-tuned Finance & Tech domain LLM."""

    def __init__(self, model_path: str, base_model: Optional[str] = None,
                 load_in_4bit: bool = True, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model(model_path, base_model, load_in_4bit)

    def _load_model(self, model_path: str, base_model: Optional[str],
                    load_in_4bit: bool):
        """Load the fine-tuned model and tokenizer."""
        print(f"\n🔄 Loading model from: {model_path}")

        # Check if LoRA adapter or full model
        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        bnb_config = None
        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        if is_lora and base_model:
            # Load base model then apply LoRA weights
            print(f"   Detected LoRA adapter. Loading base: {base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            # Load merged/full model directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.model.eval()
        print("   ✓ Model loaded successfully")

        # Print metadata if available
        metadata_path = os.path.join(model_path, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            print(f"   Base model: {meta.get('base_model', 'unknown')}")
            print(f"   Trained for: {meta.get('epochs', '?')} epochs")
            print(f"   Final loss: {meta.get('final_train_loss', '?'):.4f}")

    @torch.inference_mode()
    def generate(self, instruction: str, context: str = "",
                 max_new_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 50) -> str:
        """Generate a response for the given instruction."""
        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction.strip(),
            input=context.strip(),
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        # Decode only newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens_per_sec = len(new_tokens) / elapsed

        return response.strip(), elapsed, tokens_per_sec

    def chat(self, instruction: str, context: str = "") -> None:
        """Print formatted generation output."""
        print(f"\n{'─'*60}")
        print(f"❓ Question: {instruction}")
        if context:
            print(f"📄 Context: {context}")
        print(f"{'─'*60}")

        response, elapsed, tps = self.generate(instruction, context)

        print(f"💬 Response:\n{response}")
        print(f"\n⏱️  {elapsed:.2f}s | {tps:.1f} tokens/sec")


def run_demo(model: DomainLLM) -> None:
    """Run all demo questions and save outputs."""
    print("\n" + "=" * 60)
    print("  DEMO: Finance & Tech Domain LLM")
    print("=" * 60)

    results = []
    for i, (question, domain) in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n[{i}/{len(DEMO_QUESTIONS)}] [{domain.upper()}]")
        response, elapsed, tps = model.generate(question)

        print(f"Q: {question}")
        print(f"A: {response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"   ⏱️  {elapsed:.2f}s | {tps:.1f} tok/s")

        results.append({
            "domain": domain,
            "question": question,
            "answer": response,
            "time_seconds": round(elapsed, 2),
            "tokens_per_second": round(tps, 1),
        })

    # Save results
    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/demo_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Demo complete. Results saved → {output_file}")


def interactive_chat(model: DomainLLM) -> None:
    """Launch an interactive CLI chat loop."""
    print("\n" + "=" * 60)
    print("  Finance & Tech Domain LLM — Interactive Chat")
    print("  Type 'quit' to exit | 'help' for example questions")
    print("=" * 60)

    examples = [
        "What is EBITDA?",
        "Explain Kubernetes briefly.",
        "What is a P/E ratio?",
        "How does CI/CD work?",
        "What are derivatives in finance?",
    ]

    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break
            if user_input.lower() == "help":
                print("Example questions:")
                for ex in examples:
                    print(f"  • {ex}")
                continue

            model.chat(user_input)

        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the fine-tuned Finance & Tech LLM")
    parser.add_argument("--model_path", type=str, default="outputs/final_model",
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (needed if model_path is a LoRA adapter only)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to run (skips interactive mode)")
    parser.add_argument("--context", type=str, default="",
                        help="Optional input context for the prompt")
    parser.add_argument("--demo", action="store_true",
                        help="Run all demo questions and save outputs")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive chat (default if no prompt/demo)")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization (use for CPU/MPS)")
    args = parser.parse_args()

    # Check model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        print("   Run training first: python scripts/train.py")
        return

    # Load model
    model = DomainLLM(
        model_path=args.model_path,
        base_model=args.base_model,
        load_in_4bit=not args.no_4bit,
    )

    if args.demo:
        run_demo(model)
    elif args.prompt:
        model.chat(args.prompt, args.context)
    else:
        interactive_chat(model)


if __name__ == "__main__":
    main()
