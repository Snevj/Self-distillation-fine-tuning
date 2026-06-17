import os
import re
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
# Explicitly import from the official TRL experimental branch
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

# 1. Dashboard Mapping Tracking Configuration
os.environ["TRACKIO_PROJECT"] = "my-sdpo-alignment"
os.environ["TRACKIO_SPACE_ID"] = "Snevj/my-training-dashboard"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

max_seq_length = 2048

# FIXED: Points directly to the top-level Kaggle working directory where SFT saved the weights
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/kaggle/working/outputs-sft", 
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
# 3. Configure the model adapters for Reinforcement Learning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 4. Define a basic Verifiable Math Reward Function for GSM8K
def accuracy_reward_fn(completions, answer, **kwargs):
    rewards = []
    for completion, gt_answer in zip(completions, answer):
        # Extract the last numerical sequence block found in the assistant text
        numbers = re.findall(r'\d+', completion)
        if numbers and numbers[-1] == str(gt_answer):
            rewards.append(1.0) # Correct match reward
        else:
            rewards.append(0.0) # Error penalty
    return rewards

# 5. Load the small reinforcement learning dataset tracking framework
dataset = load_dataset("gsm8k", "main", split="train[:1000]")

# Map column names to fit TRL requirements ('prompt' is required by SDPO)
dataset = dataset.rename_column("question", "prompt")

# 6. Initialize the SDPO Configuration Block
training_args = SDPOConfig(
    output_dir = "outputs-sdpo",
    max_steps = 150,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    logging_steps = 1,
    optim = "adamw_8bit",
    # SDPO Technical parameters tuned for a single 16GB GPU instance
    distillation_weight = 1.0,
    distillation_mode = "topk_logits",
    distillation_topk = 100,
    teacher_model_kind = "ema",
    use_vllm = False, # Keeps generation local inside standard RAM memory bounds
    report_to = "trackio",
    run_name = "sdpo_alignment_run_01"
)

# 7. Initialize the Official TRL SDPOTrainer
trainer = SDPOTrainer(
    model = model,
    tokenizer = tokenizer,
    args = training_args,
    train_dataset = dataset,
    reward_funcs = [accuracy_reward_fn],
)

# Launch the RL Self-Distillation Optimization execution block
print("Starting main SDPO training loop...")
trainer.train()

# Save the final optimized adapter weights 
model.save_pretrained("outputs-sdpo")

# 8. Automated GGUF Quantization Step
print("Merging weights and quantizing model into 4-bit GGUF block...")
model.save_pretrained_gguf(
    "outputs_sdpo_gguf", 
    tokenizer, 
    quantization_method = "q4_k_m"
)
print("Pipeline complete! Transferred model binary is ready.")