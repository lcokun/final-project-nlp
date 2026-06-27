import os

import matplotlib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_CHECKPOINT = "xlm-roberta-base"
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
SEED       = 42

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE, "Data", "balanced_corpus_fixed.csv")
OUTPUT_DIR  = os.path.join(BASE, "models", "xlm_roberta_bilingual")
RESULTS_DIR = os.path.join(BASE, "results")
HF_REPO     = "lcokun/toxic-comment-xlm-roberta-bilingual"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
df = pd.read_csv(DATA_PATH).dropna(subset=["preprocessed_text"])
print(f"Dataset: {df.shape[0]} rows | {df['toxic'].value_counts().to_dict()}")
print(df.groupby(["lang", "toxic"]).size())

X = df["preprocessed_text"].tolist()
y = df["toxic"].tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN
    )


def make_dataset(texts, labels):
    ds = Dataset.from_dict({"text": texts, "label": labels})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


train_ds = make_dataset(X_train, y_train)
val_ds   = make_dataset(X_val,   y_val)
test_ds  = make_dataset(X_test,  y_test)

train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=2
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    seed=SEED,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Training...")
trainer.train()

print("Evaluating on test set...")
preds_output = trainer.predict(test_ds)
y_pred = np.argmax(preds_output.predictions, axis=-1)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

print(f"\nTest Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Non-Toxic", "Toxic"],
    yticklabels=["Non-Toxic", "Toxic"],
    ax=ax,
)
ax.set_title("Confusion Matrix — XLM-RoBERTa Bilingual")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cm_xlm_roberta_bilingual.png"), dpi=150)
plt.close()

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"Pushed to HuggingFace Hub: {HF_REPO}")

print("Done.")
