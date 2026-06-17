
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch 
import os

# Groups your SFT, DPO, and SDPO runs into one dashboard
os.environ["TRACKIO_PROJECT"] = "my-sdpo-alignment"


# Trackio will automatically create a live, free web dashboard here!
os.environ["TRACKIO_SPACE_ID"] = "Snevj/my-training-dashboard"


# Loading the 4-bit quantized base model (LLaMA 3 8B in this case, but you can use any supported model)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
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

# Loading SFT dataset 
# dataset = load_dataset("my_dataset", split = "train")

# Setting up the SFT Trainer
sft_trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    # train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs-sft",

        report_to = "trackio", 
        run_name = "sft_model_run_01",
    ),
)

# sft_trainer.train()
# model.save_pretrained("sft_model_checkpoint")