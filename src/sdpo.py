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
from transformers import GenerationConfig

max_seq_length = 2048

# SFT saves inside the repo dir, not /kaggle/working root
SFT_OUTPUT = "/kaggle/working/Self-distillation-fine-tuning/outputs-sft"

# 1. Load the model (Loads SFT base + existing LoRA adapters automatically)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = SFT_OUTPUT,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

# NOTE: Removed manual `get_peft_model` to prevent double-adapter nesting 
# and eliminate the SDPO student/teacher architecture warnings.
FastLanguageModel.for_training(model) 

# ── Generation Config (Fixes warning spam & enables variety) ──────────────────
generation_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=True,             # Critical: forces variation across the 4 generations
    temperature=0.7,            # Higher value breaks deterministic/flat rewards
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# ── Reward function ────────────────────────────────────────────────────────────
def accuracy_reward_fn(completions, answer, **kwargs):
    rewards = []
    for completion, gt_answer in zip(completions, answer):
        if isinstance(completion, list):
            text = " ".join(
                m["content"] for m in completion
                if isinstance(m, dict) and m.get("role") == "assistant"
            ) or " ".join(str(m) for m in completion)
        else:
            text = str(completion)
            
        # Extract GSM8K ground truth target number
        gt_match = re.search(r'####\s*(-?\d+)', str(gt_answer))
        gt_num = gt_match.group(1) if gt_match else str(gt_answer).strip()
        
        # Grab the very last number from the model's response chain
        numbers = re.findall(r'-?\d+', text)
        final_pred = numbers[-1] if numbers else None
        
        rewards.append(1.0 if final_pred == gt_num else 0.0)
    return rewards

# ── Dataset ────────────────────────────────────────────────────────────────────
raw_dataset = load_dataset("gsm8k", "main", split="train[:1000]")

def format_prompt(examples):
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
    per_device_train_batch_size = 1,       
    gradient_accumulation_steps = 8,       
    learning_rate = 5e-5,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    
    # SDPO Parameters
    num_generations = 4,                   
    distillation_weight = 1.0,             
    distillation_mode = "topk_logits",
    distillation_topk = 100,
    teacher_model_kind = "ema",
    use_vllm = False,
    success_reward_threshold = 0.1,       # Lower threshold to trigger self-distillation easily
    report_to = "trackio",
    run_name = "sdpo_alignment_run_01",
)

# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = SDPOTrainer(
    model = model,
    processing_class = tokenizer,
    args = training_args,
    train_dataset = dataset,
    reward_funcs = [accuracy_reward_fn],
    generation_config = generation_config, # Forces clean generation parsing
)

print("Starting SDPO training loop...")
trainer.train()

model.save_pretrained("outputs-sdpo")
print("SDPO training complete! Weights saved to 'outputs-sdpo'.")

# ── GGUF export ───────────────────────────────────────────────────────────────
print("Quantizing to GGUF (q4_k_m)...")
model.save_pretrained_gguf("outputs_sdpo_gguf", tokenizer, quantization_method="q4_k_m")
print("Pipeline complete!");