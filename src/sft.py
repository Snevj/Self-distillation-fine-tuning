import os
import torch

# ── Hugging Face auth ──────────────────────────────────────────────────────────
_hf_token = os.environ.get("HF_TOKEN")
if not _hf_token:
    raise EnvironmentError(
        "HF_TOKEN secret not found. "
        "Go to Kaggle → Add-ons → Secrets, add a secret named 'HF_TOKEN' "
        "with your token from https://huggingface.co/settings/tokens (needs Write access), "
        "then enable it for this notebook and re-run."
    )
os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
os.environ["HF_TOKEN"] = _hf_token
# ──────────────────────────────────────────────────────────────────────────────

os.environ["UNSLOTH_FORCE_FP16"] = "1"
# Reduces CUDA memory fragmentation — recommended by PyTorch for OOM errors
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRACKIO_PROJECT"] = "my-sdpo-alignment"
os.environ["TRACKIO_SPACE_ID"] = "Snevj/my-training-dashboard"

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth",  # Saves ~30% VRAM vs standard checkpointing
)

print("Loading SFT instruction dataset...")
raw_sft_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train[:20000]")

def formatting_prompts_func(examples):
    texts = []
    for instruction, output in zip(examples["question"], examples["answer"]):
        messages = [
            {"role": "user", "content": f"Question: {instruction}\nAnswer: Let's think step by step."},
            {"role": "assistant", "content": output}
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

dataset = raw_sft_dataset.map(formatting_prompts_func, batched=True)
print("SFT Dataset ready and formatted!")

sft_trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = True,
        per_device_train_batch_size = 1,       # Reduced from 2 → frees ~1.5GB
        gradient_accumulation_steps = 8,       # Increased from 4 → keeps effective batch = 8
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "trackio",
        dataloader_pin_memory = False,         # Frees a small chunk of pinned CPU memory
    ),
)

sft_trainer.train()

model.save_pretrained("outputs-sft")
print("SFT Training complete and weights saved to 'outputs-sft'!")