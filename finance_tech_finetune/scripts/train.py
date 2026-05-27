"""
train.py
=========
Fine-tunes LLaMA-3 or Mistral using QLoRA (4-bit quantization + LoRA adapters)
with HuggingFace PEFT and TRL's SFTTrainer for domain-specific Finance & Tech tasks.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --model mistralai/Mistral-7B-Instruct-v0.3 --epochs 5
"""

import argparse
import os
import sys
import json
import time
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, DatasetDict

# ── Reproducibility ──────────────────────────────────────────
set_seed(42)


# ── Configuration Dataclass ──────────────────────────────────
@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_token: Optional[str] = None
    train_file: str = "data/combined_dataset.jsonl"
    val_split: float = 0.1
    max_seq_length: int = 2048
    output_dir: str = "outputs/checkpoints"
    final_model_dir: str = "outputs/final_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    report_to: str = "none"
    run_name: str = "finance-tech-llm-v1"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_new_tokens: int = 512
    push_to_hub: bool = False
    hub_model_id: str = "your-username/finance-tech-llm"


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load config from YAML file and override with dataclass defaults."""
    config = TrainingConfig()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        model_cfg = raw.get("model", {})
        quant_cfg = raw.get("quantization", {})
        lora_cfg = raw.get("lora", {})
        data_cfg = raw.get("dataset", {})
        train_cfg = raw.get("training", {})
        out_cfg = raw.get("output", {})

        config.base_model = model_cfg.get("base_model", config.base_model)
        config.hf_token = model_cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        config.train_file = data_cfg.get("train_file", config.train_file)
        config.val_split = data_cfg.get("val_split", config.val_split)
        config.max_seq_length = data_cfg.get("max_seq_length", config.max_seq_length)
        config.lora_r = lora_cfg.get("r", config.lora_r)
        config.lora_alpha = lora_cfg.get("lora_alpha", config.lora_alpha)
        config.lora_dropout = lora_cfg.get("lora_dropout", config.lora_dropout)
        config.output_dir = train_cfg.get("output_dir", config.output_dir)
        config.num_train_epochs = train_cfg.get("num_train_epochs", config.num_train_epochs)
        config.per_device_train_batch_size = train_cfg.get("per_device_train_batch_size", config.per_device_train_batch_size)
        config.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", config.gradient_accumulation_steps)
        config.learning_rate = train_cfg.get("learning_rate", config.learning_rate)
        config.bf16 = train_cfg.get("bf16", config.bf16)
        config.fp16 = train_cfg.get("fp16", config.fp16)
        config.logging_steps = train_cfg.get("logging_steps", config.logging_steps)
        config.save_steps = train_cfg.get("save_steps", config.save_steps)
        config.eval_steps = train_cfg.get("eval_steps", config.eval_steps)
        config.report_to = train_cfg.get("report_to", config.report_to)
        config.run_name = train_cfg.get("run_name", config.run_name)
        config.final_model_dir = out_cfg.get("final_model_dir", config.final_model_dir)
        config.push_to_hub = out_cfg.get("push_to_hub", config.push_to_hub)
        config.hub_model_id = out_cfg.get("hub_model_id", config.hub_model_id)
        print(f"✓ Loaded config from {config_path}")
    return config


def check_gpu() -> str:
    """Check and report GPU availability."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🖥️  Device: Apple MPS (bitsandbytes 4-bit not supported — using float16)")
        return "mps"
    else:
        print("⚠️  No GPU detected — training on CPU will be extremely slow")
        return "cpu"


def load_tokenizer(model_name: str, hf_token: Optional[str]) -> AutoTokenizer:
    """Load and configure tokenizer."""
    print(f"\n🔤 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    # Set pad token (required for batched training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Required for SFTTrainer
    print(f"   Vocab size: {tokenizer.vocab_size:,}")
    print(f"   Pad token: {tokenizer.pad_token}")
    return tokenizer


def load_model(config: TrainingConfig, device: str) -> AutoModelForCausalLM:
    """Load base model with 4-bit quantization."""
    print(f"\n🤖 Loading model: {config.base_model}")

    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("   Quantization: 4-bit QLoRA (NF4)")

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device != "cuda" else None,
        token=config.hf_token,
        trust_remote_code=True,
    )

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params / 1e9:.2f}B")
    return model


def setup_lora(model: AutoModelForCausalLM, config: TrainingConfig) -> AutoModelForCausalLM:
    """Apply LoRA adapters to the model."""
    print(f"\n🔩 Applying LoRA adapters (r={config.lora_r}, alpha={config.lora_alpha})")

    # Auto-detect target modules based on model architecture
    if "llama" in config.base_model.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    elif "mistral" in config.base_model.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]  # Fallback

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable:,} ({100*trainable/total:.2f}% of total)")
    print(f"   Target modules: {target_modules}")
    return model


def load_training_data(config: TrainingConfig) -> DatasetDict:
    """Load and split the training dataset."""
    print(f"\n📊 Loading dataset: {config.train_file}")

    if not os.path.exists(config.train_file):
        print("  ⚠️  Combined dataset not found. Running prepare_dataset.py first...")
        import subprocess
        subprocess.run([sys.executable, "scripts/prepare_dataset.py"], check=True)

    dataset = load_dataset("json", data_files=config.train_file, split="train")
    split = dataset.train_test_split(test_size=config.val_split, seed=42)
    dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})

    print(f"   Train: {len(dataset_dict['train'])} samples")
    print(f"   Val:   {len(dataset_dict['validation'])} samples")
    return dataset_dict


def train(config: TrainingConfig, dataset: DatasetDict, model, tokenizer):
    """Run the fine-tuning training loop."""
    print("\n🚀 Starting fine-tuning...")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Batch size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} grad accumulation")
    print(f"   Effective batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"   Learning rate: {config.learning_rate}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.report_to,
        run_name=config.run_name,
        dataloader_num_workers=4,
        group_by_length=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        remove_unused_columns=False,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
    )

    # Train
    start_time = time.time()
    trainer_output = trainer.train()
    elapsed = time.time() - start_time

    print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")
    print(f"   Final train loss: {trainer_output.training_loss:.4f}")

    # Save final model
    print(f"\n💾 Saving final model → {config.final_model_dir}")
    os.makedirs(config.final_model_dir, exist_ok=True)
    trainer.save_model(config.final_model_dir)
    tokenizer.save_pretrained(config.final_model_dir)

    # Save training metadata
    metadata = {
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "final_train_loss": trainer_output.training_loss,
        "training_time_minutes": round(elapsed / 60, 2),
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
    }
    with open(os.path.join(config.final_model_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {config.final_model_dir}/training_metadata.json")

    # Push to Hub
    if config.push_to_hub and config.hub_model_id:
        print(f"\n☁️  Pushing to HuggingFace Hub: {config.hub_model_id}")
        trainer.push_to_hub(config.hub_model_id)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Finance & Tech domain")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Override base model path/name")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  Domain-Specific LLM Fine-Tuning — Finance & Tech")
    print("  Assignment: Mahesh Sale")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    if args.model:
        config.base_model = args.model
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.output_dir:
        config.output_dir = args.output_dir

    # Check GPU
    device = check_gpu()

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.final_model_dir, exist_ok=True)

    # Load components
    tokenizer = load_tokenizer(config.base_model, config.hf_token)
    model = load_model(config, device)
    model = setup_lora(model, config)

    # Load data
    dataset = load_training_data(config)

    # Train
    trainer = train(config, dataset, model, tokenizer)

    print("\n" + "=" * 60)
    print("  Fine-Tuning Complete!")
    print(f"  Model saved at: {config.final_model_dir}")
    print("  Next: python scripts/inference.py --model_path", config.final_model_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
