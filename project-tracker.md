# Self-Distillation Fine-Tuning Pipeline — Project Tracker

## Status

```
Status: Execution
Last updated: June 14, 2026
Current stage: Setting up the environment
Next step: checking the availability for SDPO
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
- [ ] Set up environment (trl, transformers, peft, bitsandbytes, accelerate, HF token)
- [ ] Export/log traces from CrewAI document-processing agent (task, messages, tool calls, errors, corrections)
- [ ] Write evaluator prompt + `evaluate_trace()` function
- [ ] Run evaluator on traces, filter to ones with a concrete feedback point
- [ ] Convert filtered traces into SDPO rows (prompt + privileged_context)
- [ ] Configure LoRA (r=8) + SDPOConfig
- [ ] Run small test (max_steps=10-20) to verify pipeline runs end-to-end
- [ ] Design custom reward function (beyond naive length-based reward)
- [ ] Run full training with logging enabled
- [ ] Evaluate adapter on held-out traces — does it act on the evaluator's hints?

### Wrap-up
- [ ] Document results (loss/reward curves, before/after comparison)
- [ ] Push adapter to Hugging Face Hub (optional)
- [ ] Write resume bullet for "Self-Distillation Fine-Tuning Pipeline" project

---

## Notes
- Reference script: [SDPO self-distillation gist](https://gist.github.com/burtenshaw/61c4a4b367409bf7a6000cc0ec54c483)
- Base models used in reference: gemma-4-26B-A4B-it (evaluator), gemma-4-12B-it (student)
- TRL's `SDPOTrainer` is recent — verify exact API/parameter names against current TRL docs before coding
