import os
import numpy as np
import pandas as pd
import optuna
from datasets import load_dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy
from peft import get_peft_model, LoraConfig, TaskType  # Assuming 'peft' is the library you're using for LoRA
import evaluate

os.environ['WANDB_DISABLED'] = 'true'

dataset = load_dataset("Melricflash/CW_MedAbstracts")
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
valid_dataset = tokenized_datasets["test"].shuffle(seed=42)

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    lora_alpha = trial.suggest_int('lora_alpha', 2, 32, log=True)
    rank = trial.suggest_int('rank', 8, 64, log=True)

    model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=5)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=lora_alpha, lora_dropout=0.1)
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
    )

    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
        return {"f1": f1["f1"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial

print(f"Value: {trial.value}")
print("Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
