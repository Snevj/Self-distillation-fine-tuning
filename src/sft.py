
from unsloth import FastLanguageModel, get_chat_template, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset
import torch 
import os

import torch
import os

# FORCE fallback to standard fp16/float16 math since P100 doesn't natively parse bfloat16
os.environ["UNSLOTH_FORCE_FP16"] = "1"
# Groups SFT, DPO, and SDPO runs into one dashboard which is automatically created on Hugging Face Spaces
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

# Loading SFT dataset 
# dataset = load_dataset("my_dataset", split = "train")
# Loading a high-quality math reasoning dataset for SFT
print("Loading SFT instruction dataset...")
raw_sft_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train[:20000]") # Using the first 20k rows for faster free-tier execution

# Defined the formatting function that wraps data in the chat template
def formatting_prompts_func(examples):
    instructions = examples["question"]
    outputs      = examples["answer"]
    texts = []
    
    for instruction, output in zip(instructions, outputs):
        # Structured the turn data into the expected ChatML/OpenAI schema
        messages = [
            {"role": "user", "content": f"Question: {instruction}\nAnswer: Let's think step by step."},
            {"role": "assistant", "content": output}
        ]
        # Convert the dictionary list into a single tokenizable string raw text format
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(formatted_text)
        
    return { "text" : texts }

# Mapping the dataset to generate the combined 'text' column required by SFTTrainer
dataset = raw_sft_dataset.map(formatting_prompts_func, batched=True)
print("SFT Dataset ready and formatted!")

# Import SFTConfig from TRL
from trl import SFTConfig

# Setting up the SFT Trainer
sft_trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,  # FIXED: Enables optimized sequence packing for padding-free mode
    args = TrainingArguments(
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