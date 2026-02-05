# Testing Agent Fine-Tuning Plan

**Goal:** Fine-tune DeepSeek-Coder 6.7B for pytest test generation to improve TestingCrew Pass@1 and Pass@3 on CodeMind tasks.

**Strategy:** 2-stage training (KodCode → TestGenEval) + optional Stage C (failure traces).

---

## Quick + High ROI Summary

| Stage | Data | Config |
|-------|------|--------|
| **A** | KodCode 30k–50k deduped | tokenized (prompt+completion) filter, r=16, alpha=32, dropout 0.05, 1 epoch |
| **B** | TestGenEval train | full test file generation only, 0.5–1 epoch, LR 1e-4 |
| **C** (optional) | 50–200 failure→fix traces | LR 5e-5, few hundred steps |

**Prompt constraint:** `pytest only, no hypothesis, no unittest`

---

## 1. Architecture Overview

```
finetuning/
├── PLAN.md                 # This document
├── requirements.txt        # TRL, PEFT, datasets, etc.
├── config.py               # Model, LoRA, training hyperparams
├── data/
│   ├── __init__.py
│   ├── kodcode.py         # Load & format KodCode-V1-SFT-R1
│   ├── testgeneval.py     # Load & format TestGenEval
│   ├── failure_traces.py  # (Optional) Format TestingCrew failure→fix
│   └── prompt.py           # Shared prompt template
├── train.py               # Main training entry (Stage A or B)
├── train_stage_a.py       # Stage A: KodCode
├── train_stage_b.py       # Stage B: TestGenEval (loads Stage A adapter)
└── export_adapter.py      # Save/merge adapters for Ollama or HF
```

---

## 2. Prompt Template (Tests-Only)

**Input format (same for both datasets):**

```
### Task: Write pytest tests

### Code Under Test:
{paste solution/code}

### Constraints:
pytest only, no hypothesis, no unittest. Edge cases, parametrize, no explanation.

### Existing tests:  (optional, for completion tasks only)
{preamble if TestGenEval completion}
```

**Output format:** Raw `test_*.py` content only — no markdown, no explanation.

**Rationale:** TestingCrew extracts code from markdown blocks, but training the model to emit raw test files reduces token waste and aligns with "runnable output."

---

## 3. Data Pipeline

### 3.1 KodCode-V1-SFT-R1 (Stage A)

| Source Column | Our Mapping |
|---------------|-------------|
| `solution` | Code Under Test |
| `test` | Target output (test file) |
| `question` | (Optional) extra context; we can omit for tests-only |

**Filtering:**
- Use `train` split only (268k rows)
- Inspect `dataset.column_names` and map to expected fields (HF versions vary)
- Assert required columns exist; raise KeyError if not
- Filter: `r1_correctness == True` (verified passing) if column exists
- Filter: tokenized length of (prompt + completion) ≤ seq_len (not just test length)
- Subset: 30k–50k deduped samples (hash of normalized solution)

**Format conversion:**
```python
# Pseudocode
def format_kodcode(row):
    prompt = PROMPT_TEMPLATE.format(
        code=row["solution"],
        constraints="pytest, edge cases, parametrize, no explanation",
        existing_tests=""  # KodCode is generation, not completion
    )
    return {"prompt": prompt, "completion": row["test"]}
```

**Note:** KodCode tests use `from solution import ...` — we keep that; TestingCrew runs tests with `cwd` set to target file dir, so imports may need adaptation. Stage B (TestGenEval) addresses real repo imports.

---

### 3.2 TestGenEval (Stage B)

| Source Column | Our Mapping |
|---------------|-------------|
| `code_src` | Code Under Test |
| `test_src` | Full test file (target) |

**Settings (fastest ROI):**
- Always full generation: prompt includes code → output full test file
- No preamble/last slicing — mixing completion modes makes training noisier
- Inspect `column_names` and assert `code_src`, `test_src` exist

**Format:**
```python
def format_testgeneval(row):
    code = row["code_src"]
    prompt = PROMPT_TEMPLATE.format(code=code, existing_tests="")
    return {"prompt": prompt, "completion": row["test_src"]}
```

**Splits:** train (15.4k), dev (210) — use train for Stage B, hold out dev for eval.

---

### 3.3 Failure Traces — Stage C (Optional, Best ROI)

**Input:** `code + generated_test + pytest_failure_log`  
**Output:** `corrected_test_file`

**Source:** TestingCrew runs — log (code, test, failure_output, fixed_test) when a refinement succeeds.

**Implementation:** 
- Add logging in `TestingCrew.run_iteration` when `iteration > 1` and `all_passed`
- Export to JSONL: `{code, failed_test, failure_log, fixed_test}`
- Format as SFT pairs, 50–200 examples
- **Stage C config:** very small LR (5e-5), short run (few hundred steps)

---

## 4. Model & Training Config

### Base Model
- **HuggingFace:** `deepseek-ai/deepseek-coder-6.7b-instruct` (or base if instruct not needed)
- **Deploy:** Train/eval using HF checkpoints first. Later: decide whether to deploy via vLLM / llama.cpp / Ollama conversion. Ollama + GGUF doesn't natively apply LoRA — conversion is a separate workflow. Don't get blocked on Ollama export.

### QLoRA / PEFT
| Param | Value |
|-------|-------|
| 4-bit NF4 | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=...)` |
| LoRA r | 16 |
| LoRA alpha | 32 |
| dropout | 0.05 |
| target_modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (include MLP for better code adaptation) |

### Training
| Param | Stage A | Stage B |
|-------|---------|---------|
| LR | 2e-4 | 1e-4 (lower, adapt) |
| seq_len | 2048 | 2048 (4096 if needed) |
| batch_size | 4–8 (grad accum 2–4) | 4 |
| epochs | 1 | 0.5–1 |
| steps | 2k–6k (subset) | ~4k (full train) |
| warmup | 0.03 | 0.03 |

---

## 5. Training Flow

### Stage A
1. Load KodCode train split
2. Filter + subset 20k–50k
3. Format with prompt template
4. Train 1 epoch, save adapter to `outputs/stage_a/`

### Stage B
1. Load base model + Stage A adapter
2. Load TestGenEval train
3. Format with prompt template
4. Train 0.5–1 epoch, save adapter to `outputs/stage_b/`
5. (Optional) Merge adapters for single model

### Export
- Save merged adapter to HuggingFace
- Deploy with HF/vLLM first for ROI
- Ollama/GGUF conversion is a later step if needed (not straightforward — LoRA merge required)

---

## 6. Evaluation Integration

**Metrics:** Pass@1, Pass@3 (unchanged)

**Where to measure:**
1. **CodeMind internal:** `evaluation/benchmarks/self_correction.json` (30 tasks)
2. **TestGenEval dev:** Held-out 210 — no train leakage

**How:**
- `eval_self_correction.py` already uses `TestingCrew(model=...)` 
- Add `model="path/to/adapter"` or `model="hf-repo/testing-agent"` when using fine-tuned model
- Run before/after fine-tuning to report delta

---

## 7. Dependencies

```
# finetuning/requirements.txt
torch>=2.1.0
transformers>=4.37.0
datasets>=2.16.0
accelerate>=0.26.0
peft>=0.8.0
bitsandbytes>=0.42.0
trl>=0.7.0
```

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| KodCode `from solution import` doesn't match CodeMind layout | Stage B (TestGenEval) uses real repo structure |
| Seq overflow | Filter by tokenized (prompt+completion) length |
| Overfitting on small subset | 1 epoch, early stop on dev loss |
| Ollama compatibility | HF first; Ollama conversion is separate workflow (merge LoRA → base, then convert) |

---

## 9. Implementation Order

1. **config.py** — Central config
2. **data/prompt.py** — Prompt template
3. **data/kodcode.py** — KodCode loader + formatter
4. **train.py** — Generic SFT loop (TRL SFTTrainer)
5. **train_stage_a.py** — Wire KodCode, run Stage A
6. **data/testgeneval.py** — TestGenEval loader
7. **train_stage_b.py** — Wire TestGenEval, load Stage A adapter
8. **export_adapter.py** — Merge + save
9. **TestingCrew integration** — Use fine-tuned model in eval
10. **(Optional)** failure_traces.py + pipeline
