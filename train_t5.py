# train_t5.py
import os
import json
import time
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import argparse

def load_jsonl(path):
    """Load JSONL file with format: {'zh': '...', 'en': '...', ...}"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append({'zh': item['zh'], 'en': item['en']})
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='t5-small')
    parser.add_argument('--data_path', type=str, default='data/train_10k.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs/t5')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    print(f"🚀 Fine-tuning {args.model_name} for Zh→En NMT")
    
    # === 1. Load training data ===
    raw_train = load_jsonl(args.data_path)
    print(f"✅ Loaded {len(raw_train)} training examples from {args.data_path}")

    # === 2. Try to load validation data ===
    valid_path = os.path.join(os.path.dirname(args.data_path), 'valid.jsonl')
    if os.path.exists(valid_path):
        raw_valid = load_jsonl(valid_path)
        print(f"✅ Loaded {len(raw_valid)} validation examples from {valid_path}")
        has_valid = True
    else:
        raw_valid = None
        has_valid = False
        print("⚠️  No valid.jsonl found. Training without validation.")

    # === 3. Build datasets ===
    def add_prefix(examples):
        inputs = [f"translate Chinese to English: {zh}" for zh in examples['zh']]
        return {'input': inputs, 'target': examples['en']}
    
    train_dataset = Dataset.from_list(raw_train).map(add_prefix, batched=True)
    if has_valid:
        valid_dataset = Dataset.from_list(raw_valid).map(add_prefix, batched=True)

    # === 4. Tokenizer & Model ===
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # === 5. Tokenize function ===
    max_len = 128
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['input'],
            max_length=max_len,
            truncation=True,
            padding=False  # dynamic padding via data collator
        )
        labels = tokenizer(
            examples['target'],
            max_length=max_len,
            truncation=True,
            padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['zh', 'en', 'input', 'target']
    )
    if has_valid:
        tokenized_valid = valid_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['zh', 'en', 'input', 'target']
        )

    # === 6. Training setup ===
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    logging_steps=100,
    save_strategy="epoch",
    # evaluation_strategy="epoch" if has_valid else "no",  
    # eval_steps=None,
    # load_best_model_at_end=False,
    # greater_is_better=False,
    # metric_for_best_model="eval_loss" if has_valid else None,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=True,
)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100  # ignore padding in loss
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid if has_valid else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # === 7. Train ===
    start_time = time.time()
    train_result = trainer.train()
    total_time = time.time() - start_time

    # === 8. Save custom log (for your report) ===
    log_history = trainer.state.log_history
    train_losses = [log['loss'] for log in log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]

    log_data = {
        'model_type': 't5',
        'base_model': args.model_name,
        'train_data': args.data_path,
        'valid_data': valid_path if has_valid else None,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation': args.grad_accum,
        'learning_rate': args.lr,
        'train_losses': train_losses,
        'eval_losses': eval_losses if has_valid else None,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_eval_loss': eval_losses[-1] if eval_losses else None,
        'total_training_time_sec': total_time,
        'num_train_examples': len(raw_train),
        'num_valid_examples': len(raw_valid) if has_valid else 0,
    }

    with open(os.path.join(args.output_dir, 'train_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    # === 9. Save final model ===
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Training completed! Logs saved to {args.output_dir}/train_log.json")

if __name__ == "__main__":
    main()