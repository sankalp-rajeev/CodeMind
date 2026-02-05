# ==========================================
# 🌙 CODEGEMMA/GEMMA 3 OVERNIGHT EVALUATION
# ==========================================
# INSTRUCTIONS:
# 1. Open a NEW Google Colab Notebook.
# 2. Runtime -> Change Runtime Type -> T4 GPU (Standard/Free).
# 3. Copy-paste these cells and Run All.
# 4. Go to sleep. 😴

# CELL 1: Install Dependencies
# ------------------------------------------
# Install Unsloth & dependencies for 4-bit loading
import os
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install datasets codebleu

# CELL 2: Mount Drive & Config
# ------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Path to your stored adapters (Adjust if you named it differently)
# We look for the folder structure. If you zip it, unzip it first.
ADAPTER_ZIP_PATH = "/content/drive/MyDrive/gemma3-12b-adapters.zip" 
EXTRACT_PATH = "/content/adapters"

if os.path.exists(ADAPTER_ZIP_PATH):
    print("📦 Found Adapter Zip. Extracting...")
    !unzip -q {ADAPTER_ZIP_PATH} -d {EXTRACT_PATH}
    # Adjust path if zip contained a subfolder
    if "gemma3-12b-testgen-lora" in os.listdir(EXTRACT_PATH):
        ADAPTER_PATH = f"{EXTRACT_PATH}/gemma3-12b-testgen-lora"
    else:
        ADAPTER_PATH = EXTRACT_PATH
else:
    # Fallback to direct folder if exists
    ADAPTER_PATH = "/content/drive/MyDrive/gemma3-12b-testgen-lora"

print(f"✅ Adapter Path set to: {ADAPTER_PATH}")


# CELL 3: Load & Split Data (REPRODUCING THE SPLIT)
# ------------------------------------------
from datasets import load_dataset, concatenate_datasets

print("📚 Loading Datasets...")
ds_coderm = load_dataset("cryptoguard/code-rm-unit-test", split="train")
ds_humaneval = load_dataset("openai_humaneval", split="test")
ds_mbpp = load_dataset("mbpp", split="test")

# Combine
combined = concatenate_datasets([ds_coderm, ds_humaneval, ds_mbpp])

# Re-Split to isolate the exact SAME test set
# Random seed 3407 (Must match training notebook)
combined = combined.shuffle(seed=3407) 
total = len(combined)
train_end = int(0.90 * total)
val_end = int(0.95 * total)

# The "Held-Out" Test Set (Last 5%)
test_dataset = combined.select(range(val_end, total))
print(f"📊 Dataset Ready. Testing on {len(test_dataset)} samples.")


# CELL 4: Evaluation Metrics & Logic
# ------------------------------------------
import torch
import re
from unsloth import FastLanguageModel

def extract_code(text):
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].strip()
    return text.strip()

def run_evaluation(model, tokenizer, dataset, desc="Eval", limit=None):
    """Run model on dataset and return list of results"""
    model.eval()
    results = []
    
    # Run on a subset if limit provided, else full test set
    target_data = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    print(f"🚀 [{desc}] Starting evaluation on {len(target_data)} samples...")

    for i, sample in enumerate(target_data):
        # Handle different column names
        source_code = sample.get('code', sample.get('prompt', sample.get('text', '')))
        
        prompt = f"Write pytest unit tests for:\n```python\n{source_code}\n```"
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=512, 
                temperature=0.0, # Greedy/Deterministic for fair benchmark
                use_cache=True
            )
        
        raw_output = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        generated_code = extract_code(raw_output)
        
        results.append({
            "source": source_code,
            "generated": generated_code,
            "raw": raw_output
        })
        
        if (i+1) % 10 == 0:
            print(f"   Processed {i+1}/{len(target_data)}")
            
    return results

def compute_metrics(results):
    """Simple syntax/validity check"""
    valid = 0
    has_assert = 0
    total = len(results)
    
    for r in results:
        code = r['generated']
        # 1. Syntax Check
        try:
            compile(code, '<string>', 'exec')
            valid += 1
        except:
            pass
        # 2. Assert Check
        if "assert" in code:
            has_assert += 1
            
    return {
        "total": total,
        "syntax_valid_pct": (valid/total)*100,
        "has_assertions_pct": (has_assert/total)*100
    }

# CELL 5: RUN BASELINE EVALUATION
# ------------------------------------------
print("\n" + "="*50)
print("🏗️ PHASE 1: BASELINE (Gemma 3 12B - Base)")
print("="*50)

# Load Base Model Only
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-3-12b-it",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Run Eval
baseline_output = run_evaluation(model, tokenizer, test_dataset, desc="Baseline")
baseline_metrics = compute_metrics(baseline_output)
print(f"📉 Baseline Metrics: {baseline_metrics}")

# Clear VRAM to be safe
del model
torch.cuda.empty_cache()


# CELL 6: RUN FINE-TUNED EVALUATION
# ------------------------------------------
print("\n" + "="*50)
print("🚀 PHASE 2: FINE-TUNED (With Adapters)")
print("="*50)

# Load Base + Adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_PATH, # Loads base + adapters automatically
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Run Eval
ft_output = run_evaluation(model, tokenizer, test_dataset, desc="Fine-tuned")
ft_metrics = compute_metrics(ft_output)
print(f"📈 Fine-tuned Metrics: {ft_metrics}")


# CELL 7: SAVE REPORT TO DRIVE
# ------------------------------------------
import json
import pandas as pd

report = {
    "baseline": baseline_metrics,
    "finetuned": ft_metrics,
    "delta_validity": ft_metrics['syntax_valid_pct'] - baseline_metrics['syntax_valid_pct'],
    "delta_assertions": ft_metrics['has_assertions_pct'] - baseline_metrics['has_assertions_pct']
}

print("\n" + "="*50)
print(f"🎉 FINAL REPORT")
print(f"Validity Improvement: {report['delta_validity']:+.1f}%")
print(f"Assertion Improvement: {report['delta_assertions']:+.1f}%")
print("="*50)

# Save RAW data for manual inspection
with open("/content/drive/MyDrive/full_eval_results.json", "w") as f:
    json.dump({"baseline_samples": baseline_output, "ft_samples": ft_output}, f)

print("✅ Saved full logs to: /content/drive/MyDrive/full_eval_results.json")
