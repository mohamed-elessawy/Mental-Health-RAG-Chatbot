import os
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,)
from sklearn.metrics import f1_score

os.makedirs("models/distilbert", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

assert os.path.exists("data/train.parquet"), \
    "Preprocessed data not found. Run eda.py first."

label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
num_labels = len(label_names)

df_train = pd.read_parquet("data/train.parquet")
df_val = pd.read_parquet("data/val.parquet")

dataset = DatasetDict({"train":Dataset.from_pandas(df_train[["text", "label"]]),
    "validation": Dataset.from_pandas(df_val[["text", "label"]]),})

print(f"Labels: {label_names}")
print(f"Training samples: {len(dataset['train'])}")

# some configurations for training
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 2e-5
OUTPUT_DIR = "models/distilbert"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels") # for Hugging Face Trainer compatibility
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) #--> convert to PyTorch tensors.

label_counts  = np.bincount(dataset["train"]["label"]) # it counts how many samples belong to each emotion class in the training dataset.
class_weights = torch.tensor(1.0 / label_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * num_labels

# move class weights to the same device as the model (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class_weights = class_weights.to(device)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={i: l for i, l in enumerate(label_names)},
    label2id={l: i for i, l in enumerate(label_names)},
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds, average="macro")}


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=0.1, # for transformer stability
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True, # important to restores best checkpoint
    metric_for_best_model="f1",
    logging_dir="outputs/logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
)


trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nStarting training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved → {OUTPUT_DIR}")