
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import GeminiModel

# Load environment variables (Looking for GOOGLE_API_KEY)
load_dotenv()

# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️ WARNING: GOOGLE_API_KEY not found in .env. DeepEval metrics using Gemini will fail.")

# Configuration
BASE_MODEL_ID = "google/gemma-3-12b-it" 
# Specific artifact path requested by user
ADAPTER_PATH = r"c:\Umich\Projects\CodeMind\gemma3-12b-testgen-lora-20260204T055813Z-3-001\gemma3-12b-testgen-lora"

# Initialize Gemini Judge
# Using gemini-1.5-flash as a cost-effective and fast judge
custom_judge = GeminiModel(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

print("🚀 Loading Base Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)

print(f"🔗 Loading LoRA Adapters from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("✅ Model Loaded!")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

def generate_test(code_snippet):
    """Generates a pytest unit test for the given code snippet."""
    prompt = f"<start_of_turn>user\nWrite pytest unit tests for:\n```python\n{code_snippet}\n```<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # Basic cleanup to extract code block if present
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
        
    return response

# Define Metrics using Gemini Judge

# 1. Answer Relevancy (Does the test relate to the input?)
relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=custom_judge)

# 2. Custom G-Eval for Code Correctness
correctness_metric = GEval(
    name="Pytest Validity",
    criteria="Determine whether the 'actual output' is a valid, runnable pytest code block that accurately tests the logic in 'input'. Check for syntax errors, proper imports, and meaningful assertions.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=custom_judge
)

# 3. Prompt Alignment (Did it write a test?)
alignment_metric = GEval(
    name="Prompt Alignment",
    criteria="Check if the output follows the instruction to 'Write pytest unit tests'. It should not be an explanation, but actual code.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=custom_judge
)

# Sample Data
samples = [
    """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
    """
]

test_cases = []

print("\n🧪 Running Evaluation on Samples...")
for i, sample_code in enumerate(samples):
    print(f"\nSample {i+1} Generation:")
    generated_test = generate_test(sample_code)
    print(generated_test[:200] + "...") # Print preview
    
    test_case = LLMTestCase(
        input=sample_code,
        actual_output=generated_test,
        retrieval_context=[sample_code] # Context is the code itself
    )
    test_cases.append(test_case)
    
    # Clean up memory to avoid OOM
    import gc
    gc.collect()
    torch.cuda.empty_cache()


print("\n⚖️  Running DeepEval Metrics (Judge: Gemini)...")
results = evaluate(test_cases, [relevancy_metric, correctness_metric, alignment_metric])

# Save Results
import json
import datetime

output_dir = "finetuning/eval_results"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"deepeval_local_{timestamp}.json")

# Simple serialization helper
def serialize_metric(m):
    return {
        "name": m.__class__.__name__,
        "score": m.score,
        "reason": getattr(m, "reason", "No explanation provided")
    }

final_data = []
for r in results:
    final_data.append({
        "input": r.input,
        "actual_output": r.actual_output,
        "success": r.success,
        "metrics": [serialize_metric(m) for m in r.metrics]
    })

with open(output_path, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")
