import os
import torch

# ── Hugging Face auth ──────────────────────────────────────────────────────────
# Kaggle: Notebook → Add-ons → Secrets → add secret named HF_TOKEN (Write access),
# then toggle it ON for this notebook.
# Must happen FIRST — before any other import — so trackio/huggingface_hub see it.
_hf_token = os.environ.get("HF_TOKEN")
if not _hf_token:
    raise EnvironmentError(
        "HF_TOKEN secret not found. "
        "Go to Kaggle → Add-ons → Secrets, add a secret named 'HF_TOKEN' "
        "with your token from https://huggingface.co/settings/tokens (needs Write access), "
        "then enable it for this notebook and re-run."
    )
# Set both env vars huggingface_hub checks for full compatibility
os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
os.environ["HF_TOKEN"] = _hf_token
# ──────────────────────────────────────────────────────────────────────────────

# FORCE fallback to standard fp16/float16 math since P100 doesn't natively support bfloat16
os.environ["UNSLOTH_FORCE_FP16"] = "1"
# Groups SFT, DPO, and SDPO runs into one dashboard automatically created on Hugging Face Spaces
os.environ["TRACKIO_PROJECT"] = "my-sdpo-alignment"
# Trackio will automatically create a live, free web dashboard here!
os.environ["TRACKIO_SPACE_ID"] = "Snevj/my-training-dashboard"

from unsloth import FastLanguageModel, get_chat_template, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Loading the 4-bit quantized base model (LLaMA 3 8B)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
)

# Adding the LoRA adapter to the model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# Loading a high-quality math reasoning dataset for SFT
print("Loading SFT instruction dataset...")
raw_sft_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train[:20000]")

def formatting_prompts_func(examples):
    instructions = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        messages = [
            {"role": "user", "content": f"Question: {instruction}\nAnswer: Let's think step by step."},
            {"role": "assistant", "content": output}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(formatted_text)
    return { "text" : texts }

dataset = raw_sft_dataset.map(formatting_prompts_func, batched=True)
print("SFT Dataset ready and formatted!")

# Setting up the SFT Trainer
sft_trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
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
    ),
)

# Executing the training loop
sft_trainer.train()

# Saving the completed SFT weights to a local folder
model.save_pretrained("outputs-sft")
print("SFT Training complete and weights saved to 'outputs-sft'!")