"""
prepare_dataset.py
==================
Merges finance and tech JSONL datasets, validates entries,
formats them into the instruction-following prompt template,
and saves a combined training-ready dataset.

Usage:
    python scripts/prepare_dataset.py
"""

import json
import os
import random
from pathlib import Path
from typing import Optional
from datasets import Dataset, DatasetDict


# ── Prompt Template (Alpaca-style) ──────────────────────────
PROMPT_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """\
### Instruction:
{instruction}

### Response:
{output}"""


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping line {line_num} in {filepath}: {e}")
    return records


def validate_record(record: dict, source: str) -> Optional[dict]:
    """Validate that required fields exist and are non-empty."""
    required = ["instruction", "output"]
    for field in required:
        if field not in record or not str(record[field]).strip():
            print(f"  ✗ Skipping record — missing or empty '{field}' in {source}")
            return None
    # Ensure 'input' field exists (can be empty)
    if "input" not in record:
        record["input"] = ""
    return record


def format_prompt(record: dict) -> str:
    """Format a record into the instruction prompt template."""
    if record.get("input", "").strip():
        return PROMPT_TEMPLATE.format(
            instruction=record["instruction"].strip(),
            input=record["input"].strip(),
            output=record["output"].strip(),
        )
    else:
        return PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=record["instruction"].strip(),
            output=record["output"].strip(),
        )


def prepare_datasets(
    finance_path: str = "data/finance_dataset.jsonl",
    tech_path: str = "data/tech_dataset.jsonl",
    output_path: str = "data/combined_dataset.jsonl",
    val_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Load, validate, merge, format, and split datasets.
    Returns a HuggingFace DatasetDict with train/validation splits.
    """
    print("=" * 60)
    print("  Finance & Tech Domain — Dataset Preparation")
    print("=" * 60)

    # ── Load Raw Data ────────────────────────────────────────
    print(f"\n📂 Loading finance dataset from: {finance_path}")
    finance_records = load_jsonl(finance_path)
    print(f"   ✓ Loaded {len(finance_records)} finance records")

    print(f"\n📂 Loading tech dataset from: {tech_path}")
    tech_records = load_jsonl(tech_path)
    print(f"   ✓ Loaded {len(tech_records)} tech records")

    # ── Validate & Tag ───────────────────────────────────────
    print("\n🔍 Validating records...")
    validated_finance = []
    for r in finance_records:
        validated = validate_record(r, "finance")
        if validated:
            validated["domain"] = "finance"
            validated_finance.append(validated)

    validated_tech = []
    for r in tech_records:
        validated = validate_record(r, "tech")
        if validated:
            validated["domain"] = "tech"
            validated_tech.append(validated)

    print(f"   ✓ Finance: {len(validated_finance)} valid records")
    print(f"   ✓ Tech:    {len(validated_tech)} valid records")

    # ── Merge & Shuffle ──────────────────────────────────────
    all_records = validated_finance + validated_tech
    random.seed(seed)
    random.shuffle(all_records)

    print(f"\n🔀 Combined & shuffled: {len(all_records)} total records")

    # ── Format Prompts ───────────────────────────────────────
    print("\n✏️  Formatting into instruction-following prompts...")
    formatted = []
    for record in all_records:
        formatted.append({
            "text": format_prompt(record),
            "instruction": record["instruction"],
            "input": record.get("input", ""),
            "output": record["output"],
            "domain": record["domain"],
        })

    # ── Save Combined JSONL ──────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in formatted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n💾 Saved combined dataset → {output_path}")

    # ── Create HuggingFace Dataset ───────────────────────────
    hf_dataset = Dataset.from_list(formatted)

    # Train/Validation split
    split = hf_dataset.train_test_split(test_size=val_split, seed=seed)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    # ── Print Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  Total samples  : {len(formatted)}")
    print(f"  Train samples  : {len(dataset_dict['train'])}")
    print(f"  Val samples    : {len(dataset_dict['validation'])}")
    print(f"  Finance records: {len(validated_finance)}")
    print(f"  Tech records   : {len(validated_tech)}")

    # Domain distribution
    domains = [r["domain"] for r in formatted]
    fin_count = domains.count("finance")
    tech_count = domains.count("tech")
    print(f"\n  Domain Distribution:")
    print(f"    Finance : {fin_count} ({fin_count/len(domains)*100:.1f}%)")
    print(f"    Tech    : {tech_count} ({tech_count/len(domains)*100:.1f}%)")

    # Sample output preview
    print("\n  Sample formatted prompt preview:")
    print("-" * 60)
    preview = formatted[0]["text"][:400]
    print(preview + "..." if len(formatted[0]["text"]) > 400 else preview)
    print("=" * 60)

    return dataset_dict


if __name__ == "__main__":
    dataset = prepare_datasets(
        finance_path="data/finance_dataset.jsonl",
        tech_path="data/tech_dataset.jsonl",
        output_path="data/combined_dataset.jsonl",
        val_split=0.1,
        seed=42,
    )
    print("\n✅ Dataset preparation complete!")
    print("   Next step: python scripts/train.py --config configs/training_config.yaml")
