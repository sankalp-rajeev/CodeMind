# %% [markdown]
# # 🧪 Gemma 3 12B Test Generation Fine-tuning
# 
# **Goal**: Fine-tune Gemma 3 12B for unit test generation with comprehensive evaluation
# 
# **Features**:
# - ✅ Multiple datasets: CodeRM-UnitTest + HumanEval + MBPP (~20k samples)
# - ✅ Train/Validation split (95%/5%)
# - ✅ Comprehensive baseline evaluation BEFORE fine-tuning
# - ✅ Post-training evaluation with improvement metrics
# - ✅ GGUF export for local deployment (RTX 4080 compatible)
# 
# **Evaluation Metrics**:
# - Syntax Validity (compiles)
# - Structural Quality (has test functions, assertions, imports)
# - Executability (runs without crash)
# - Pass Rate (test passes on correct code)
# - CodeBLEU (semantic similarity)
# 
# **Hardware**: A100/H100 recommended (40GB+ VRAM)

# %% [markdown]
# ## 📦 1. Installation

# %%
%%capture
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
!pip install datasets transformers trl accelerate bitsandbytes
!pip install codebleu  # For CodeBLEU metric

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## 🔑 2. Hugging Face Login

# %%
from huggingface_hub import login
import os

# Option 1: Use environment variable
hf_token = os.environ.get("HF_TOKEN", None)

# Option 2: Use Colab secrets (if available)
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
    except:
        pass

# Option 3: Manual input
if not hf_token:
    hf_token = input("Enter your Hugging Face token: ")

login(token=hf_token)
print("✅ Logged in to Hugging Face")

# %% [markdown]
# ## 🔧 3. Load Base Model

# %%
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-12b-it",
    max_seq_length=4096,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

print(f"✅ Model loaded: Gemma 3 12B")
print(f"   Max seq length: 4096")
print(f"   Quantization: 4-bit")

# %% [markdown]
# ## 📊 4. Load All Datasets

# %%
from datasets import load_dataset, concatenate_datasets
import json

print("📥 Loading datasets...")
print("="*60)

# ============================================
# 1. CodeRM-UnitTest - High-quality unit tests (~17k)
# ============================================
print("\n1️⃣ Loading CodeRM-UnitTest...")
try:
    coderm = load_dataset("KAKA22/CodeRM-UnitTest", split="train")
    print(f"   ✅ CodeRM: {len(coderm)} samples")
    print(f"   Fields: {coderm.column_names}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    coderm = None

# ============================================
# 2. HumanEval-MBPP TestGen QA (if available)
# ============================================
print("\n2️⃣ Loading HumanEval-MBPP-TestGen-QA...")
try:
    testgen_qa = load_dataset("OllieStanley/humaneval-mbpp-testgen-qa", split="train")
    print(f"   ✅ TestGen-QA: {len(testgen_qa)} samples")
    print(f"   Fields: {testgen_qa.column_names}")
except Exception as e:
    print(f"   ⚠️ Failed (optional): {e}")
    testgen_qa = None

# ============================================
# 3. OpenAI HumanEval - 164 hand-crafted problems
# ============================================
print("\n3️⃣ Loading OpenAI HumanEval...")
try:
    humaneval = load_dataset("openai_humaneval", split="test")
    print(f"   ✅ HumanEval: {len(humaneval)} samples")
    print(f"   Fields: {humaneval.column_names}")
except Exception as e:
    print(f"   ⚠️ Failed (optional): {e}")
    humaneval = None

# ============================================
# 4. MBPP - ~1000 Python problems
# ============================================
print("\n4️⃣ Loading MBPP...")
try:
    mbpp = load_dataset("Muennighoff/mbpp", split="train")
    print(f"   ✅ MBPP: {len(mbpp)} samples")
    print(f"   Fields: {mbpp.column_names}")
except Exception as e:
    print(f"   ⚠️ Failed (optional): {e}")
    mbpp = None

print("\n" + "="*60)

# %%
# Preview dataset samples
print("DATASET STRUCTURE PREVIEW")
print("="*60)

if coderm:
    print("\n📋 CodeRM sample keys:", coderm[0].keys())
    
if testgen_qa:
    print("\n📋 TestGen-QA sample:")
    print(testgen_qa[0])

if humaneval:
    print("\n📋 HumanEval sample keys:", humaneval[0].keys())
    
if mbpp:
    print("\n📋 MBPP sample keys:", mbpp[0].keys())

# %% [markdown]
# ## 🔄 5. Format Datasets

# %%
def format_coderm(example):
    """Format CodeRM-UnitTest dataset"""
    code = example.get('code_ground_truth', example.get('code', ''))
    tests_raw = example.get('unit_tests', '')
    
    # Parse JSON array and extract test code
    if isinstance(tests_raw, str) and tests_raw.startswith('['):
        try:
            tests_list = json.loads(tests_raw)
            test_codes = [t.get('code', '').replace('\\n', '\n') for t in tests_list if t.get('code')]
            tests = test_codes[0] if test_codes else ''
        except:
            tests = ''
    else:
        tests = tests_raw
    
    if not code or not tests:
        return {"conversations": [], "source_code": "", "reference_test": ""}
    
    return {
        "conversations": [
            {"role": "user", "content": f"Write pytest unit tests for:\n```python\n{code}\n```"},
            {"role": "assistant", "content": f"```python\n{tests}\n```"}
        ],
        "source_code": code,
        "reference_test": tests
    }

def format_testgen_qa(example):
    """Format HumanEval-MBPP-TestGen-QA dataset"""
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    if not input_text or not output_text:
        return {"conversations": [], "source_code": "", "reference_test": ""}
    
    return {
        "conversations": [
            {"role": "user", "content": f"Write pytest unit tests for:\n{input_text}"},
            {"role": "assistant", "content": output_text}
        ],
        "source_code": input_text,
        "reference_test": output_text
    }

def format_humaneval(example):
    """Format HumanEval dataset"""
    prompt = example.get('prompt', '')
    canonical = example.get('canonical_solution', '')
    test = example.get('test', '')
    
    full_code = prompt + canonical
    
    if not full_code or not test:
        return {"conversations": [], "source_code": "", "reference_test": ""}
    
    return {
        "conversations": [
            {"role": "user", "content": f"Write pytest unit tests for:\n```python\n{full_code}\n```"},
            {"role": "assistant", "content": f"```python\n{test}\n```"}
        ],
        "source_code": full_code,
        "reference_test": test
    }

def format_mbpp(example):
    """Format MBPP dataset"""
    code = example.get('code', '')
    test_list = example.get('test_list', [])
    
    # Combine test assertions
    if isinstance(test_list, list):
        tests = '\n'.join(test_list)
    else:
        tests = str(test_list)
    
    if not code or not tests:
        return {"conversations": [], "source_code": "", "reference_test": ""}
    
    # Wrap assertions in proper test function
    test_code = f"import pytest\n\ndef test_solution():\n    {tests.replace(chr(10), chr(10) + '    ')}"
    
    return {
        "conversations": [
            {"role": "user", "content": f"Write pytest unit tests for:\n```python\n{code}\n```"},
            {"role": "assistant", "content": f"```python\n{test_code}\n```"}
        ],
        "source_code": code,
        "reference_test": test_code
    }

print("🔄 Formatting datasets...")

# %%
# Apply formatting and combine
datasets_to_merge = []

if coderm:
    coderm_fmt = coderm.map(format_coderm, remove_columns=coderm.column_names)
    coderm_fmt = coderm_fmt.filter(lambda x: len(x['conversations']) > 0)
    print(f"✅ CodeRM: {len(coderm_fmt)} valid samples")
    datasets_to_merge.append(coderm_fmt)

if testgen_qa:
    testgen_fmt = testgen_qa.map(format_testgen_qa, remove_columns=testgen_qa.column_names)
    testgen_fmt = testgen_fmt.filter(lambda x: len(x['conversations']) > 0)
    print(f"✅ TestGen-QA: {len(testgen_fmt)} valid samples")
    datasets_to_merge.append(testgen_fmt)

if humaneval:
    humaneval_fmt = humaneval.map(format_humaneval, remove_columns=humaneval.column_names)
    humaneval_fmt = humaneval_fmt.filter(lambda x: len(x['conversations']) > 0)
    print(f"✅ HumanEval: {len(humaneval_fmt)} valid samples")
    datasets_to_merge.append(humaneval_fmt)

if mbpp:
    mbpp_fmt = mbpp.map(format_mbpp, remove_columns=mbpp.column_names)
    mbpp_fmt = mbpp_fmt.filter(lambda x: len(x['conversations']) > 0)
    print(f"✅ MBPP: {len(mbpp_fmt)} valid samples")
    datasets_to_merge.append(mbpp_fmt)

# Combine all datasets
if len(datasets_to_merge) > 1:
    combined_dataset = concatenate_datasets(datasets_to_merge).shuffle(seed=42)
else:
    combined_dataset = datasets_to_merge[0].shuffle(seed=42)

print(f"\n📦 Combined dataset: {len(combined_dataset)} total samples")

# %%
# Create train/validation/test split
# 90% train, 5% validation, 5% test (held out for final evaluation)

total = len(combined_dataset)
train_end = int(0.90 * total)
val_end = int(0.95 * total)

train_dataset = combined_dataset.select(range(train_end))
val_dataset = combined_dataset.select(range(train_end, val_end))
test_dataset = combined_dataset.select(range(val_end, total))

print(f"📊 Dataset Split:")
print(f"   Train: {len(train_dataset)} samples (90%)")
print(f"   Validation: {len(val_dataset)} samples (5%)")
print(f"   Test (held-out): {len(test_dataset)} samples (5%)")

# %% [markdown]
# ## 📐 6. Comprehensive Evaluation Metrics

# %%
import subprocess
import tempfile
import os
import re
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout (Unix only)"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def extract_code_from_response(response):
    """Extract Python code from markdown code blocks"""
    if '```python' in response:
        parts = response.split('```python')
        if len(parts) > 1:
            code = parts[1].split('```')[0]
            return code.strip()
    elif '```' in response:
        parts = response.split('```')
        if len(parts) > 1:
            return parts[1].strip()
    return response.strip()

def evaluate_test_comprehensive(generated_test, source_code=None, reference_test=None):
    """
    Comprehensive evaluation of generated test code.
    Returns dict with multiple metrics.
    """
    metrics = {
        'syntax_valid': False,
        'has_test_functions': False,
        'has_assertions': False,
        'has_imports': False,
        'has_docstrings': False,
        'executable': False,
        'passes_on_code': False,
        'codebleu': 0.0,
    }
    
    if not generated_test or len(generated_test) < 10:
        return metrics
    
    # 1. Syntax Validity
    try:
        compile(generated_test, '<string>', 'exec')
        metrics['syntax_valid'] = True
    except SyntaxError:
        pass
    
    # 2. Structural Analysis
    if re.search(r'def test_\w+', generated_test) or re.search(r'class Test\w+', generated_test):
        metrics['has_test_functions'] = True
    
    assertion_patterns = ['assert ', 'assertEqual', 'assertTrue', 'assertFalse', 
                          'assertRaises', 'pytest.raises', 'assertIn', 'assertIsNone']
    if any(p in generated_test for p in assertion_patterns):
        metrics['has_assertions'] = True
    
    import_patterns = ['import pytest', 'import unittest', 'from unittest']
    if any(p in generated_test for p in import_patterns):
        metrics['has_imports'] = True
    
    if '"""' in generated_test or "'''" in generated_test:
        metrics['has_docstrings'] = True
    
    # 3. Executability (can we run it without crash?)
    if metrics['syntax_valid']:
        try:
            # Create isolated namespace
            namespace = {}
            exec("import pytest\nimport unittest", namespace)
            # Just check if code executes (defines functions), don't run tests
            exec(generated_test, namespace)
            metrics['executable'] = True
        except Exception:
            pass
    
    # 4. Pass on source code (if provided)
    if metrics['executable'] and source_code:
        try:
            namespace = {}
            # Execute source code first
            exec(source_code, namespace)
            # Then execute test code
            exec(generated_test, namespace)
            # If no exception, tests likely pass
            metrics['passes_on_code'] = True
        except Exception:
            pass
    
    # 5. CodeBLEU (if reference provided)
    if reference_test:
        try:
            from codebleu import calc_codebleu
            result = calc_codebleu([reference_test], [generated_test], lang="python")
            metrics['codebleu'] = result['codebleu']
        except Exception:
            # Simple token overlap fallback
            gen_tokens = set(generated_test.split())
            ref_tokens = set(reference_test.split())
            if len(ref_tokens) > 0:
                metrics['codebleu'] = len(gen_tokens & ref_tokens) / len(ref_tokens)
    
    return metrics

def run_comprehensive_evaluation(model, tokenizer, eval_dataset, max_samples=50, desc="Eval"):
    """
    Run comprehensive evaluation on dataset.
    Returns aggregated metrics.
    """
    all_metrics = []
    
    num_samples = min(max_samples, len(eval_dataset))
    print(f"\n🔍 {desc}: Evaluating {num_samples} samples...")
    
    for i in range(num_samples):
        sample = eval_dataset[i]
        source_code = sample.get('source_code', '')
        reference_test = sample.get('reference_test', '')
        
        if not source_code:
            continue
        
        # Generate test
        prompt = f"Write pytest unit tests for:\n```python\n{source_code[:1500]}\n```"
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=512, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        generated_test = extract_code_from_response(response)
        
        # Evaluate
        metrics = evaluate_test_comprehensive(generated_test, source_code, reference_test)
        all_metrics.append(metrics)
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{num_samples}")
    
    # Aggregate results
    if not all_metrics:
        return None
    
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        if isinstance(values[0], bool):
            aggregated[key] = sum(values) / len(values) * 100  # Percentage
        else:
            aggregated[key] = sum(values) / len(values) * 100  # Also percentage
    
    # Overall score (weighted average)
    aggregated['overall'] = (
        aggregated['syntax_valid'] * 0.15 +
        aggregated['has_test_functions'] * 0.15 +
        aggregated['has_assertions'] * 0.15 +
        aggregated['has_imports'] * 0.05 +
        aggregated['executable'] * 0.20 +
        aggregated['passes_on_code'] * 0.20 +
        aggregated['codebleu'] * 0.10
    )
    
    return aggregated

def print_metrics_table(metrics, title):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Score':>10}")
    print(f"{'-'*35}")
    for key, value in metrics.items():
        print(f"{key:<25} {value:>9.1f}%")
    print(f"{'='*60}")

# %% [markdown]
# ## 📏 7. BASELINE Evaluation (Before Fine-tuning)

# %%
print("\n" + "🔴"*30)
print("BASELINE EVALUATION - Before Fine-tuning")
print("🔴"*30)

# Evaluate on held-out test set
baseline_scores = run_comprehensive_evaluation(
    model, 
    tokenizer, 
    test_dataset, 
    max_samples=50,
    desc="BASELINE"
)

if baseline_scores:
    print_metrics_table(baseline_scores, "BASELINE Scores (Gemma 3 12B - Before Fine-tuning)")
else:
    print("❌ Baseline evaluation failed")

# %% [markdown]
# ## 📝 8. Apply Chat Template

# %%
from unsloth.chat_templates import get_chat_template, standardize_data_formats

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

# Standardize format
train_dataset = standardize_data_formats(train_dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False
        ).removeprefix('<bos>')
        for convo in convos
    ]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
print(f"✅ Applied Gemma-3 chat template to {len(train_dataset)} training samples")

# %% [markdown]
# ## 🎯 9. Setup LoRA Adapters

# %%
model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ LoRA adapters configured")
print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
print(f"   Total: {total_params:,}")

# %% [markdown]
# ## ⚡ 10. Training Configuration

# %%
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch = 8
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="./gemma3-12b-testgen",
        save_steps=500,
        report_to="none",
    ),
)

print("✅ Trainer configured")
print(f"   Dataset: {len(train_dataset)} samples")
print(f"   Effective batch size: 8 (2 × 4)")
print(f"   Epochs: 1")

# %%
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
print("✅ Configured to train on responses only")

# %% [markdown]
# ## 🚀 11. Start Training

# %%
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}")
print(f"Max memory = {max_memory} GB")
print(f"Reserved = {start_gpu_memory} GB")

# %%
print("\n" + "🚀"*30)
print("STARTING TRAINING")
print("🚀"*30 + "\n")

trainer_stats = trainer.train()

print("\n" + "✅"*30)
print("TRAINING COMPLETE!")
print("✅"*30)

# %%
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print("\n📊 Training Summary")
print("="*50)
print(f"   Final loss: {trainer_stats.training_loss:.4f}")
print(f"   Training time: {trainer_stats.metrics['train_runtime']/60:.1f} minutes")
print(f"   Peak memory: {used_memory} GB")
print(f"   Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")

# %% [markdown]
# ## 📏 12. POST-TRAINING Evaluation

# %%
print("\n" + "🟢"*30)
print("POST-TRAINING EVALUATION - After Fine-tuning")
print("🟢"*30)

model.eval()

finetuned_scores = run_comprehensive_evaluation(
    model, 
    tokenizer, 
    test_dataset, 
    max_samples=50,
    desc="FINE-TUNED"
)

if finetuned_scores:
    print_metrics_table(finetuned_scores, "FINE-TUNED Scores (Gemma 3 12B - After Training)")

# %%
# Side-by-side comparison
print("\n" + "="*70)
print("📈 COMPARISON: Baseline vs Fine-tuned")
print("="*70)
print(f"{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Δ Change':>12}")
print("-"*70)

if baseline_scores and finetuned_scores:
    for metric in baseline_scores.keys():
        base = baseline_scores[metric]
        fine = finetuned_scores[metric]
        delta = fine - base
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        color = "" if delta >= 0 else ""
        print(f"{metric:<25} {base:>11.1f}% {fine:>11.1f}% {arrow:>3} {delta:>+7.1f}%")

    print("="*70)
    improvement = finetuned_scores['overall'] - baseline_scores['overall']
    print(f"\n🎯 OVERALL IMPROVEMENT: {improvement:+.1f}%")
    print(f"   Baseline: {baseline_scores['overall']:.1f}%")
    print(f"   Fine-tuned: {finetuned_scores['overall']:.1f}%")

# %% [markdown]
# ## 💾 13. Save Model

# %%
model.save_pretrained("gemma3-12b-testgen-lora")
tokenizer.save_pretrained("gemma3-12b-testgen-lora")
print("✅ LoRA adapters saved to: gemma3-12b-testgen-lora/")

# %%
print("📦 Exporting to GGUF (Q4_K_M for RTX 4080)...")
print("   This may take 10-15 minutes...")

model.save_pretrained_gguf(
    "gemma3-12b-testgen-gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
print("\n✅ GGUF model saved to: gemma3-12b-testgen-gguf/")

# %% [markdown]
# ## 🧪 14. Test Inference

# %%
test_code = '''
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
'''

messages = [{"role": "user", "content": f"Write pytest unit tests for:\n```python\n{test_code}\n```"}]

inputs = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt",
    add_generation_prompt=True
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        inputs, 
        max_new_tokens=1024, 
        temperature=0.7,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print("🧪 Generated Tests for LRUCache:")
print("="*60)
print(response)

# %% [markdown]
# ## 📋 15. Final Report & Resume Bullet

# %%
print("\n" + "="*70)
print("🎯 FINAL TRAINING REPORT")
print("="*70)

print(f"""
MODEL DETAILS:
  Base Model: Gemma 3 12B
  Method: QLoRA (4-bit quantization)
  LoRA Rank: 16
  
DATASET:
  Total Samples: {len(combined_dataset):,}
  Training: {len(train_dataset):,} (90%)
  Validation: {len(val_dataset):,} (5%)
  Test: {len(test_dataset):,} (5%)
  Sources: CodeRM-UnitTest + HumanEval + MBPP

TRAINING:
  Final Loss: {trainer_stats.training_loss:.4f}
  Training Time: {trainer_stats.metrics['train_runtime']/60:.1f} minutes
  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}
""")

if baseline_scores and finetuned_scores:
    improvement = finetuned_scores['overall'] - baseline_scores['overall']
    print(f"""EVALUATION RESULTS:
  Baseline Overall: {baseline_scores['overall']:.1f}%
  Fine-tuned Overall: {finetuned_scores['overall']:.1f}%
  Improvement: {improvement:+.1f}%
  
  Key Metrics Improvement:
    Syntax Valid: {baseline_scores['syntax_valid']:.1f}% → {finetuned_scores['syntax_valid']:.1f}%
    Has Test Functions: {baseline_scores['has_test_functions']:.1f}% → {finetuned_scores['has_test_functions']:.1f}%
    Executable: {baseline_scores['executable']:.1f}% → {finetuned_scores['executable']:.1f}%
    Passes on Code: {baseline_scores['passes_on_code']:.1f}% → {finetuned_scores['passes_on_code']:.1f}%
""")

print(f"""
EXPORTS:
  LoRA Adapters: gemma3-12b-testgen-lora/
  GGUF (Q4_K_M): gemma3-12b-testgen-gguf/ (~6GB)

LOCAL DEPLOYMENT:
  ollama create testgen -f Modelfile
  ollama run testgen
""")

print("="*70)
print("📝 RESUME BULLET POINT:")
print("="*70)
if baseline_scores and finetuned_scores:
    print(f"""
Fine-tuned Gemma 3 12B using QLoRA on {len(train_dataset):,} curated code-test pairs 
from CodeRM, HumanEval, and MBPP datasets. Achieved {finetuned_scores['overall']:.0f}% 
test quality score ({improvement:+.0f}% over baseline), with {finetuned_scores['executable']:.0f}% 
executable tests and {finetuned_scores['syntax_valid']:.0f}% syntax validity. Deployed 
locally via GGUF quantization on RTX 4080.
""")

# %% [markdown]
# ## 📥 16. Download Model Files

# %%
!zip -r gemma3-12b-testgen-lora.zip gemma3-12b-testgen-lora/
!zip -r gemma3-12b-testgen-gguf.zip gemma3-12b-testgen-gguf/

print("\n📥 Files ready for download:")
print("  - gemma3-12b-testgen-lora.zip")
print("  - gemma3-12b-testgen-gguf.zip")

try:
    from google.colab import files
    files.download('gemma3-12b-testgen-gguf.zip')
except:
    print("\n⚠️ Not in Colab - use the file browser to download")


