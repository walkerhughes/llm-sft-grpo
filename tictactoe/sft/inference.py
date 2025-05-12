# qwen_sft_modal.py  ─ final version
"""
Run with:
  HF_TOKEN=hf_xxx WANDB_API_KEY=mywandb \
  modal run qwen_sft_modal.py::sft_train \
      --dataset-id=banachspace/tic-tac-toe-wins-Qwen2.5-3B-Instruct \
      --hf-repo-out=banachspace/Qwen-0.5B-TTT-SFT \
      --num-epochs=3
"""

import os, modal
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN  = os.getenv("HF_TOKEN", "")
WANDB_KEY = os.getenv("WANDB_API_KEY", "")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "procps")
    .pip_install(
        # order matters: unsloth first so import hooks patch transformers
        "unsloth",
        "torch>=2.1.2",
        "transformers>=4.39.0",
        "accelerate>=0.28.0",
        "bitsandbytes",
        "trl>=0.7.10",
        "datasets>=2.18.0",
        "huggingface_hub>=0.20.3",
        "wandb",
        "python-dotenv>=1.1.0",
    )
    .env({"HF_TOKEN": HF_TOKEN, "WANDB_API_KEY": WANDB_KEY})
)

app = modal.App("qwen-sft-openpipe", image=image)

# ─────────────────── Push helper ───────────────────────────────────────────
def push_adapter_to_hub(tmp_dir: str, repo_id: str):
    from huggingface_hub import HfApi, create_repo, upload_folder
    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id, repo_type="model")
        print(f"Repo {repo_id} exists.")
    except Exception:
        create_repo(repo_id, repo_type="model", token=HF_TOKEN, exist_ok=True)
    upload_folder(
        folder_path  = tmp_dir,
        repo_id      = repo_id,
        repo_type    = "model",
        token        = HF_TOKEN,
        path_in_repo = ".",
    )
    print(f"✅ Pushed adapter → https://huggingface.co/{repo_id}")

# ─────────────────── Training function ────────────────────────────────────
@app.function(gpu="A10", timeout=4 * 60 * 60)
def sft_train(
    base_model  : str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_id  : str = "banachspace/tic-tac-toe-wins-Qwen2.5-3B-Instruct",
    hf_repo_out : str = "banachspace/Qwen-0.5B-TTT-SFT",
    num_epochs  : int = 3,
    batch_size  : int = 4,
    lora_r      : int = 16,
):
    assert HF_TOKEN, "HF_TOKEN env-var must be set."

    # ── 1. import (Unsloth first) ─────────────────────────────────────────
    import unsloth                                   # ensure patches
    from unsloth import FastLanguageModel
    import torch, tempfile
    from datasets import load_dataset
    from transformers import BitsAndBytesConfig
    from trl import SFTTrainer

    # ── 2. dataset (54 chat sessions) ─────────────────────────────────────
    ds = load_dataset(dataset_id, split="train")     # already public
    print(ds)                                        # sanity (should show 54)

    # ── 3. model + tokenizer (Unsloth) ────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = base_model,
        device_map        = "auto",
        quantization_config = bnb_cfg,
        max_seq_length    = 2048,
        dtype             = torch.float16,
        trust_remote_code = True,
    )

    # ── 4. flatten chat → single string per row ───────────────────────────
    def flatten_chat(example):
        prompt = tokenizer.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,   # supervise full convo
            tokenize=False,
        )
        return {"text": prompt}

    ds = ds.map(flatten_chat, remove_columns=["messages"])
    print(ds.column_names)   # ['text']

    # ── 5. attach LoRA adapter ────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r              = lora_r,
        lora_alpha     = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
    )

    # ── 6. train with TRL SFTTrainer ──────────────────────────────────────
    trainer = SFTTrainer(
        model             = model,
        train_dataset     = ds,                 # all 54 rows
        dataset_text_field= "text",
        tokenizer         = tokenizer,
        max_seq_length    = 2048,
        args=dict(
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            fp16=True,
            gradient_checkpointing=True,
            output_dir="./outputs",
            logging_steps=10,
            save_total_limit=1,
            optim="paged_adamw_8bit",
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            report_to="wandb" if WANDB_KEY else "none",
        ),
    )
    trainer.train()
    print("✅ SFT finished")

    # ── 7. save & push adapter ────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp()
    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)
    push_adapter_to_hub(tmp_dir, hf_repo_out)

# ─────────────────── Local helper ─────────────────────────────────────────
@app.local_entrypoint()
def main():
    print(
        "Run training with:\n"
        "  modal run qwen_sft_modal.py::sft_train "
        "--dataset-id=banachspace/tic-tac-toe-wins-Qwen2.5-3B-Instruct "
        "--hf-repo-out=banachspace/Qwen-0.5B-TTT-SFT --num-epochs=3\n"
    )
