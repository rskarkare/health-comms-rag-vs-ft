"""Fine-tune SmolLM2 with LoRA on the 2,000 training question-answer pairs.

    python src/finetune.py

The trained adapter (a few MB of extra weights) is saved to models/smollm2-medquad-lora.
"""
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from common import ADAPTER_DIR, BASE_MODEL, DATA_DIR, build_messages

MAX_LENGTH = 512  # longest example in tokens; longer ones are cut
EPOCHS = 2        # passes over the training data

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(row):
    """Turn one Q&A pair into token IDs, training only on the answer part."""
    prompt = tokenizer.apply_chat_template(
        build_messages(row["Question"]), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["Answer"], add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]  # teaches the model when to stop

    input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]
    # -100 tells the loss to ignore those positions: the model is graded only on
    # predicting the answer, not on repeating the question.
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def main():
    dataset = load_dataset("json", data_files=f"{DATA_DIR}/train.jsonl", split="train")
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).float()

    # LoRA: freeze the original weights and train small add-on matrices instead.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch size of 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,                # print the loss every 10 steps
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,     # not supported on Apple GPUs
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
