"""
Fine-tuning config for Testing Agent (DeepSeek-Coder 6.7B).
QLoRA defaults per plan: fast ROI, don't waste time tuning.
"""

from dataclasses import dataclass
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent


@dataclass
class ModelConfig:
    base_model: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # Or: "deepseek-ai/deepseek-coder-6.7b-base" if instruct not needed


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Include MLP projections for better code-generation adaptation
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class StageAConfig:
    """KodCode: teach test-writing patterns."""
    dataset: str = "KodCode/KodCode-V1-SFT-R1"
    split: str = "train"
    max_samples: int = 30_000  # 20k-50k range
    num_epochs: int = 1
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 8  # Larger batch for RTX 4080
    gradient_accumulation_steps: int = 2  # Fewer syncs, same effective BS 16
    warmup_ratio: float = 0.03
    output_dir: Path = FINETUNING_ROOT / "outputs" / "stage_a"
    save_steps: int = 500


@dataclass
class StageBConfig:
    """TestGenEval: repo-real adaptation."""
    dataset: str = "kjain14/testgeneval"
    split: str = "train"
    num_epochs: float = 0.5  # 0.5-1
    learning_rate: float = 1e-4  # Lower for adaptation
    max_seq_length: int = 4096  # Real repo tests are often longer
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.03
    output_dir: Path = FINETUNING_ROOT / "outputs" / "stage_b"
    stage_a_adapter: Path | None = FINETUNING_ROOT / "outputs" / "stage_a" / "final"
    save_steps: int = 500
