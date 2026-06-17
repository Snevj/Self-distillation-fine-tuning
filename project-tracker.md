# Self-Distillation Fine-Tuning Pipeline — Project Tracker

## Status

```
Status: Execution
Last updated: June 14, 2026
Current stage: Writing all the file for local env
Next step: kaggle notebook setup for cloud training
```

---

## Implementation

### Prerequisites
- [x] Learn DPO basics (TRL docs / DPO paper)
- [x] Understand teacher/student self-distillation pattern
- [x] Understand reward functions and reward hacking
- [x] Practice LLM-as-judge pattern (structured JSON output, temperature=0)
- [x] Review TRL conventions (SFTTrainer vs DPOTrainer vs SDPOTrainer, dataset column formats)
- [ ] Set up experiment tracking (Trackio or W&B)

### Build Pipeline

#### Phase 1: Local Scripting & Workspace Setup (VS Code)
- [x] Set up local development environment (`trl`, `transformers`, `unsloth`, `datasets`, `trackio`, `.venv`).
- [x] Structure backend architecture into distinct modular stages (`train_sft.py`, `train_sdpo.py`, `Dockerfile`).
- [x] Configure memory-optimized 4-bit base model loading (`unsloth/llama-3-8b-bnb-4bit`) and LoRA target module adapters ($r=16$, $\alpha=16$).
- [x] Integrate Trackio environmental variables linked to a public Hugging Face Spaces dashboard for real-time remote telemetry.

#### Phase 2: Cloud Compute Execution (Kaggle Dual-T4 GPUs)
- [ ] Connect GitHub repository to Kaggle and configure encrypted environment variables (`HF_TOKEN`) via Kaggle Secrets.
- [ ] **Stage 1 (SFT):** Execute `train_sft.py` to stream and process 20,000 instances from `microsoft/orca-math-word-problems-200k`, formatting them into native Llama-3 chat templates to establish a format baseline inside `outputs-sft`.
- [ ] Design and implement custom programmatic verification reward function (`math_verification_reward_func`) to parse final numerical predictions using regex and evaluate them against ground-truth mathematical target parameters.
- [ ] **Stage 2 (SDPO):** Execute `train_sdpo.py` utilizing the official `trl.experimental.sdpo` engine with `teacher_model_kind="ema"` to run online reinforcement learning via test-time self-distillation on the `gsm8k` dataset.
- [ ] Stream and monitor live loss-curves, logit distribution parameters, and system tracking statistics via the public Hugging Face Space dashboard during batch background runs.

#### Phase 3: Quantization, Containerization & Production (Hybrid)
- [ ] Export fully aligned SDPO reasoning weights from Kaggle directly into a highly compressed, CPU-optimized 4-bit GGUF matrix (`q4_k_m`) via native Unsloth acceleration bindings.
- [ ] Write an unmanaged, ultra-lightweight deployment layer in `Dockerfile` utilizing the compiled `llama.cpp:server` base image to expose an OpenAI-compatible web API endpoint on port `8080`.
- [ ] Pull compiled GGUF weights onto the local Mac M2 machine and execute Docker container sanity tests (`docker run`) to verify token-generation integrity on 8GB shared RAM constraints.
- [ ] Automate image compilation workflows using GitHub Actions and push the production-ready container directly to the GitHub Container Registry (GHCR).
- [ ] Deploy the containerized API microservice to **DigitalOcean App Platform** leveraging GitHub Student Developer Credits for 24/7 high-availability cloud routing.

### Wrap-up
- [ ] Document results (loss/reward curves, before/after comparison)
- [ ] Push adapter to Hugging Face Hub (optional)
- [ ] Write resume bullet for "Self-Distillation Fine-Tuning Pipeline" project

---

## Notes
- Reference script: [SDPO self-distillation gist](https://gist.github.com/burtenshaw/61c4a4b367409bf7a6000cc0ec54c483)
- Base models used in reference: gemma-4-26B-A4B-it (evaluator), gemma-4-12B-it (student)
- TRL's `SDPOTrainer` is recent — verify exact API/parameter names against current TRL docs before coding
