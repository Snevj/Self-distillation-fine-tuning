import os
from unsloth import FastLanguageModel

from datasets import load_dataset

# 1. Trackio Configuration
#it will automatically create a live, free web dashboard on the hugging face
os.environ["TRACKIO_PROJECT"] = "sdpo-rich-feedback"
os.environ["TRACKIO_SPACE_ID"] = "Snevj/my-training-dashboard"

from datasets import Dataset

#importing the SDPO trainer and config
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
        "privileged_context": ["Your earlier answer used the wrong format."],
    }
)

training_args = SDPOConfig(
    output_dir="sdpo-model",
    distillation_mode="topk_logits",       # Explicitly select top-K logit distillation
    distillation_topk=100,                 # Required when using top-K logit distillation
    include_environment_feedback=True,     # Use dataset privileged_context for teacher reprompts
)
training_args = SDPOConfig(
    output_dir="sdpo-model",
    use_vllm=True,
    vllm_mode="server",
    teacher_model_kind="live",
    use_teacher_server=True,
    distillation_weight=1.0,
    distillation_mode="sampled_token",
)

trainer = SDPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()