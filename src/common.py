"""Shared settings and helper functions used by every script."""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADAPTER_DIR = "models/smollm2-medquad-lora"
DATA_DIR = "data/processed"
RESULTS_DIR = "results"

SYSTEM_PROMPT = (
    "You are a public health communicator. Answer the question accurately, "
    "in clear and plain language that a general audience can understand."
)


def get_device():
    """Use the Apple GPU (mps) if available, then an NVIDIA GPU, else the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_messages(question, context=None):
    """Build the chat messages. RAG passes retrieved text as `context`."""
    if context is None:
        user = question
    else:
        user = (
            "Use the reference information below to answer the question. "
            "If it does not contain the answer, say so.\n\n"
            f"Reference information:\n{context}\n\nQuestion: {question}"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def load_model(adapter_dir=None):
    """Load the base model, optionally with the fine-tuned LoRA adapter merged in."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).float()
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.to(get_device())
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, messages, max_new_tokens=256):
    """Format the messages with the chat template and generate a reply."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # always pick the most likely token: repeatable results
        repetition_penalty=1.1,   # discourages the small model from looping
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
