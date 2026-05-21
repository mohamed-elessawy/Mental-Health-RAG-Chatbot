import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,)

from sklearn.metrics import classification_report

MODEL_PATH = "models/distilbert"
MAX_LENGTH = 128

dataset = load_dataset("dair-ai/emotion")
label_names = dataset["train"].features["label"].names

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

tokenized = dataset["validation"].map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),)

predictions = trainer.predict(tokenized)
logits = predictions.predictions
preds = np.argmax(logits, axis=-1)
labels = predictions.label_ids

report = classification_report(
    labels,
    preds,
    target_names=label_names)

print("\n=== DistilBERT Classification Report ===")
print(report)

with open("outputs/distilbert_report.txt", "w") as f:
    f.write(report)

print("\nReport saved → outputs/distilbert_report.txt")