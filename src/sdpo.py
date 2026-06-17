import os
import re
from datasets import load_dataset
from unsloth import FastLanguageModel
# Import the official implementation from Hugging Face's TRL ecosystem
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer 
import torch

# Trackio will automatically pull these and update your public dashboard live
os.environ["TRACKIO_PROJECT"] = "sdpo-math-reasoning"
os.environ["TRACKIO_SPACE_ID"] = "Snevj/my-training-dashboard"

max_seq_length = 2048


# DATASET LOADING & STRUCTURING FOR SDPO

print("Loading GSM8K reasoning dataset...")
# Pull the standard grade-school math dataset directly from Hugging Face
raw_dataset = load_dataset("gsm8k", "main", split="train")

def format_gsm8k_prompt(example):
    """
    SDPO expects a column named 'prompt'. We format the question 
    so the model knows it needs to think step-by-step.
    """
    formatted_prompt = f"Question: {example['question']}\nAnswer: Let's think step by step."
    
    # GSM8K stores answers like 'The answer is 42 #### 42'. 
    # We extract the clean ground-truth number after the '####'
    clean_target = example['answer'].split('####')[-1].strip()
    
    return {
        "prompt": formatted_prompt,
        "target_answer": clean_target # We will use this in our reward function
    }

# Map the dataset to include our formatted prompt and target answers
dataset = raw_dataset.map(format_gsm8k_prompt)


# VERIFIABLE REWARD FUNCTION (The SDPO Feedback Loop)

def math_verification_reward_func(completions, target_answer, **kwargs):
    """
    SDPO uses this to check the model's self-generated trajectories.
    It looks for the final number in the model's response and checks if it matches target_answer.
    """
    rewards = []
    for completion, target in zip(completions, target_answer):
        # Use regex to grab the very last numerical value in the generated text
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", completion)
        if numbers:
            predicted_answer = numbers[-1]
            # If the model's final number matches the ground truth, give a reward of 1.0
            if predicted_answer.strip() == target.strip():
                rewards.append(1.0)
                continue
        # If it doesn't match or no number was found, reward is 0.0
        rewards.append(0.0)
    return rewards


# LOAD MODEL VIA UNSLOTH (Memory-Optimized)

print("Loading base SFT model weights...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs-sft", # This targets your completed SFT checkpoint folder
    max_seq_length = max_seq_length,
    dtype = None, # Automatically handles precision based on your hardware
    load_in_4bit = True, # Crucial to fit within Kaggle's 16GB VRAM limit
)

# Apply LoRA layers to protect base model and keep training lightweight
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)


# INITIALIZE OFFICIAL SDPO CONFIG & TRAINER

sdpo_config = SDPOConfig(
    output_dir = "outputs_sdpo_final",
    distillation_mode = "topk_logits",       # Evaluates token probability distributions
    distillation_topk = 100,
    teacher_model_kind = "ema",              # Exponential Moving Average keeps teacher VRAM minimal
    
    # Compute Parameters (Tuned explicitly for Kaggle Dual-T4 stability)
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-6,
    logging_steps = 1,
    max_steps = 150,                         # Kept bounded for free-tier wall-clock stability
    
    # Tracking
    report_to = "trackio",
    run_name = "sdpo_math_reasoning_run"
)

trainer = SDPOTrainer(
    model = model,
    config = sdpo_config,
    train_dataset = dataset,
    reward_funcs = math_verification_reward_func, # Passes the verification logic to the trainer
)

# RUN AND EXPORT

print("Starting Self-Distilled Policy Optimization...")
trainer.train()

print("Training complete! Merging and exporting to CPU-optimized GGUF format...")
# This native Unsloth feature converts your model to a 4-bit GGUF so you can run it on your 8GB Mac
#gguf(Generic GPT Unified Format) is a new open-source format for LLMs that is optimized for CPU inference. It is supported by the 
# latest versions of llama.cpp, text-generation-webui, and other popular inference engines.
model.save_pretrained_gguf("outputs_sdpo_gguf", tokenizer, quantization_method = "q4_k_m")
print("GGUF Model saved successfully in 'outputs_sdpo_gguf' directory!")