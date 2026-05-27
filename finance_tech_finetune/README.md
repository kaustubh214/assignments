# Fine-Tuning a Domain-Specific AI Model for Finance & Tech Industries

**Objective:** Fine-tune an open-source LLM (LLaMA-3 / Mistral) to master Finance and Tech domain vocabulary.

---

## Project Structure

```
finance_tech_finetune/
├── data/
│   ├── finance_dataset.jsonl       # Finance Q&A training data
│   ├── tech_dataset.jsonl          # Tech Q&A training data
│   └── combined_dataset.jsonl      # Merged dataset for training
├── scripts/
│   ├── prepare_dataset.py          # Dataset preparation & merging
│   ├── train.py                    # Fine-tuning with QLoRA (PEFT)
│   ├── inference.py                # Run the fine-tuned model
│   └── evaluate.py                 # Evaluate model performance
├── configs/
│   └── training_config.yaml        # All hyperparameters & settings
├── notebooks/
│   └── FineTuning_Demo.ipynb       # Google Colab-ready notebook
├── docs/
│   └── project_report.md           # Detailed project report
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python scripts/prepare_dataset.py
```

### 3. Fine-Tune the Model
```bash
python scripts/train.py --config configs/training_config.yaml
```

### 4. Run Inference
```bash
python scripts/inference.py --model_path outputs/final_model --prompt "What is EBITDA?"
```

### 5. Evaluate Performance
```bash
python scripts/evaluate.py --model_path outputs/final_model
```

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | `meta-llama/Meta-Llama-3-8B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3` |
| Fine-Tuning Method | QLoRA (Quantized LoRA) |
| Quantization | 4-bit (bitsandbytes) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Training Epochs | 3 |
| Batch Size | 4 (with gradient accumulation ×4) |
| Learning Rate | 2e-4 |

---

## Domain Coverage

### Finance Terms
`EBITDA` · `P/E Ratio` · `Derivatives` · `Equity Dilution` · `SIP Investment` · `Balance Sheet Analysis` · `Hedge Funds` · `Market Capitalization` · `Yield Curve` · `Amortization`

### Tech Terms
`Kubernetes` · `Microservices` · `CI/CD` · `Docker Containers` · `REST APIs` · `Vector Databases` · `Serverless` · `DevOps` · `MLOps` · `Service Mesh`

---

## Expected Outcomes

After fine-tuning, the model can:
- ✅ Answer industry-specific questions accurately
- ✅ Understand and use domain jargon fluently
- ✅ Generate professional finance/tech responses
- ✅ Summarize domain documents

---

## Hardware Requirements

| Mode | Minimum GPU VRAM |
|------|-----------------|
| Training (4-bit QLoRA) | 16 GB (A100 / V100 / T4×2) |
| Inference only | 8 GB |
| Google Colab | Free T4 (training may be slow) |

---

## References
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LLaMA-3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Mistral 7B Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
