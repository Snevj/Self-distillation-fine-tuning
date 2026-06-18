import os
import re
import torch

# ── Hugging Face auth ──────────────────────────────────────────────────────────
_hf_token = os.environ.get("HF_TOKEN")
if not _hf_token:
    raise EnvironmentError(
        "HF_TOKEN secret not found. "
        "Go to Kaggle → Add-ons → Secrets, add a secret named 'HF_TOKEN', "
        "then enable it for this notebook and re-run."
    )
os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
os.environ["HF_TOKEN"] = _hf_token
# ──────────────────────────────────────────────────────────────────────────────

os.environ["UNSLOTH_FORCE_FP16"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRACKIO_PROJECT"] = "my-sdpo-alignment"
os.environ["TRACKIO_SPACE_ID"] = "Snevj/my-training-dashboard"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

max_seq_length = 2048

# SFT saves inside the repo dir, not /kaggle/working root
SFT_OUTPUT = "/kaggle/working/Self-distillation-fine-tuning/outputs-sft"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = SFT_OUTPUT,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth",  # Saves ~30% VRAM
)

# ── Reward function ────────────────────────────────────────────────────────────
def accuracy_reward_fn(completions, answer, **kwargs):
    rewards = []
    for completion, gt_answer in zip(completions, answer):
        numbers = re.findall(r'\d+', completion)
        rewards.append(1.0 if numbers and numbers[-1] == str(gt_answer) else 0.0)
    return rewards

# ── Dataset ────────────────────────────────────────────────────────────────────
# SDPO requires 'prompt' as a list of chat messages (not a plain string)
raw_dataset = load_dataset("gsm8k", "main", split="train[:1000]")

def format_prompt(examples):
    # Wrap each question in the chat message format SDPO expects
    prompts = [
        [{"role": "user", "content": q}]
        for q in examples["question"]
    ]
    return {
        "prompt": prompts,
        "answer": examples["answer"],
    }

dataset = raw_dataset.map(format_prompt, batched=True, remove_columns=raw_dataset.column_names)

# ── SDPO Config ────────────────────────────────────────────────────────────────
training_args = SDPOConfig(
    output_dir = "outputs-sdpo",
    max_steps = 150,
    per_device_train_batch_size = 1,       # Keep at 1 to avoid OOM on T4
    gradient_accumulation_steps = 8,       # Effective batch = 8
    learning_rate = 5e-5,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    # SDPO-specific
    num_generations = 4,                   # Reduced from default (saves VRAM on T4)
    distillation_weight = 1.0,             # Pure self-distillation (no policy gradient blend)
    distillation_mode = "topk_logits",
    distillation_topk = 100,
    teacher_model_kind = "ema",
    use_vllm = False,
    report_to = "trackio",
    run_name = "sdpo_alignment_run_01",
)

# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = SDPOTrainer(
    model = model,
    tokenizer = tokenizer,
    args = training_args,
    train_dataset = dataset,
    reward_funcs = [accuracy_reward_fn],
)

print("Starting SDPO training loop...")
trainer.train()

model.save_pretrained("outputs-sdpo")
print("SDPO training complete! Weights saved to 'outputs-sdpo'.")

# ── GGUF export ───────────────────────────────────────────────────────────────
print("Quantizing to GGUF (q4_k_m)...")
model.save_pretrained_gguf("outputs_sdpo_gguf", tokenizer, quantization_method="q4_k_m")
print("Pipeline complete!")