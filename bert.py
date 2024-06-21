
from datasets import load_dataset, load_metric

dataset = load_dataset("Melricflash/CW_MedAbstracts")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
valid_split = tokenized_datasets["test"].shuffle(seed=42)

splits = train_dataset.train_test_split(train_size=0.9, shuffle=True)

train_split=splits['train']
test_split=splits['test']

from peft import LoraConfig, TaskType

rank=16
lora_alpha=32
model_name='BERT'

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=lora_alpha, lora_dropout=0.1
)

from transformers import BertForSequenceClassification

print(f"Loading model {model_name}.. with rank {rank}")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased', 
    num_labels=5
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return {"trainable": trainable_params, "all": all_param, "trainable%": 100 * trainable_params / all_param}

print_trainable_parameters(model)

from peft import get_peft_model
model = get_peft_model(model, lora_config)

print_trainable_parameters(model)

"""# Evaluate"""

import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"], "precision": precision["precision"], "recall": recall["recall"]}

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
                                 num_train_epochs=10,)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=valid_split,
    compute_metrics=compute_metrics,
)

import os
os.environ['WANDB_DISABLED'] = 'true'

trainer.train()

"""Getting Validation and Loss curves"""

import pandas as pd
training_results = pd.DataFrame(trainer.state.log_history)

training_results

train_loss = training_results['loss'].dropna()
train_eval = training_results['eval_loss'].dropna()

train_acc = training_results['eval_accuracy'].dropna()
train_f1 = training_results['eval_f1'].dropna()
train_prec = training_results['eval_precision'].dropna()
train_recall = training_results['eval_recall'].dropna()

indices = train_eval.index.tolist()

selected_loss = []
for idx in indices:
    loss_idx = idx-1
    selected_loss.append(train_loss[loss_idx])

selected_loss = pd.DataFrame(selected_loss)

print(train_eval)
print(selected_loss)

train_eval = train_eval.reset_index(drop=True)

train_eval.index = train_eval.index+1
selected_loss.index = selected_loss.index+1

selected_loss

import matplotlib.pyplot as plt
from pathlib import Path

csv_dir=f"csv/{model_name}"

Path("graphs").mkdir(exist_ok=True)
Path("confusionmatrix").mkdir(exist_ok=True)
Path(csv_dir).mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(selected_loss, label='Training Loss')
plt.plot(train_eval, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(f'graphs/BERT_loss_rank{rank}.png')  # Save the plot
plt.close()

train_curves = pd.concat([selected_loss, train_eval], axis=1)

train_curves.to_csv(f'{csv_dir}/BERT_train_curves_rank{rank}.csv', index=False)

"""Save the training metrics to a CSV"""

train_acc.reset_index(drop=True, inplace=True)
train_f1.reset_index(drop=True, inplace=True)
train_prec.reset_index(drop=True, inplace=True)
train_recall.reset_index(drop=True, inplace=True)

train_metrics = pd.concat([train_acc, train_f1, train_prec, train_recall], axis=1)

train_metrics.index = train_metrics.index+1

train_metrics

train_metrics.to_csv(f'{csv_dir}/BERT_train_metrics_rank{rank}.csv', index=False)

"""Evaluating model on test set"""

trainer.evaluate()
predictions = trainer.predict(test_split)

print(f"Predictions: {predictions}")

print("Hugging face metrics:")
print(f"F1 Score: {predictions[2]['test_f1']}")
print(f"Accuracy: {predictions[2]['test_accuracy']}")
print(f"Recall: {predictions[2]['test_recall']}")
print(f"Precision: {predictions[2]['test_precision']}")

pred_labels = predictions.predictions

pred_array = []

for i in pred_labels:
    pred_array.append(np.argmax(i, axis=-1))

# print(arr)

print(len(pred_array))

test_labels = test_split['label']

import pandas as pd

metrics=predictions[2]

df = pd.DataFrame({
    'metrics': metrics,
})

df.to_csv(f'{csv_dir}/BERT_Output_rank{rank}.csv', index=True)

test_preds = pd.DataFrame(pred_array)
test_labels = pd.DataFrame(test_labels)

test_preds_labels = pd.concat([test_preds, test_labels], axis = 1)

print(test_preds_labels)

test_preds_labels.to_csv(f'{csv_dir}/BERT_preds_labels_rank{rank}.csv', index = False)

from sklearn.metrics import classification_report

print(classification_report(test_labels, pred_array))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, pred_array)
cmdispl = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(10, 10))
cmdispl.plot(ax=ax)
plt.savefig(f'confusionmatrix/BERT_confusion_matrix_rank{rank}.png')
plt.close()

