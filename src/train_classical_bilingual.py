import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE, "Data", "balanced_corpus.csv")
MODELS_DIR  = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

log.info("Loading bilingual corpus...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["preprocessed_text"])
log.info(f"Dataset: {df.shape[0]} rows | {df['toxic'].value_counts().to_dict()}")

X = df["preprocessed_text"]
y = df["toxic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

vectorizers = {
    "tfidf": TfidfVectorizer(max_features=50000, ngram_range=(1, 1)),
    "bow":   CountVectorizer(max_features=50000, ngram_range=(1, 1)),
}

models = {
    "naive_bayes":          MultinomialNB(),
    "logistic_regression":  LogisticRegression(max_iter=1000, random_state=42),
    "svm":                  LinearSVC(max_iter=2000, random_state=42),
}

results = []

for vec_name, vectorizer in vectorizers.items():
    log.info(f"Fitting vectorizer: {vec_name}")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    vec_path = os.path.join(MODELS_DIR, f"vectorizer_{vec_name}_bilingual.joblib")
    joblib.dump(vectorizer, vec_path)
    log.info(f"Saved vectorizer -> {vec_path}")

    for model_name, model in models.items():
        log.info(f"Training: {model_name} + {vec_name}")
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train_vec, y_train)
        y_pred = model_clone.predict(X_test_vec)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec  = recall_score(y_test, y_pred, average="weighted")
        f1   = f1_score(y_test, y_pred, average="weighted")

        results.append({
            "model":      model_name,
            "vectorizer": vec_name,
            "accuracy":   round(acc,  4),
            "precision":  round(prec, 4),
            "recall":     round(rec,  4),
            "f1":         round(f1,   4),
        })

        log.info(f"  acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} + {vec_name.upper()} (BILINGUAL)")
        print(f"{'='*60}")
        print(classification_report(y_test, y_pred, target_names=["Non-Toxic", "Toxic"]))

        model_path = os.path.join(MODELS_DIR, f"{model_name}_{vec_name}_bilingual.joblib")
        joblib.dump(model_clone, model_path)
        log.info(f"  Saved model -> {model_path}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Toxic", "Toxic"],
            yticklabels=["Non-Toxic", "Toxic"],
            ax=ax
        )
        ax.set_title(f"Confusion Matrix\n{model_name} + {vec_name} (bilingual)")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        cm_path = os.path.join(RESULTS_DIR, f"cm_{model_name}_{vec_name}_bilingual.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
results_path = os.path.join(RESULTS_DIR, "model_comparison_bilingual.csv")
results_df.to_csv(results_path, index=False)

print(f"\n{'='*60}")
print("BILINGUAL MODEL COMPARISON (sorted by F1)")
print(f"{'='*60}")
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
bars = ax.bar(x, results_df["f1"], color=["#4C72B0", "#DD8452", "#55A868",
                                           "#C44E52", "#8172B2", "#937860"])
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{r['model'].replace('_', ' ').title()}\n({r['vectorizer'].upper()})"
     for _, r in results_df.iterrows()],
    fontsize=9
)
ax.set_ylabel("Weighted F1 Score")
ax.set_title("Model Comparison — Bilingual Classical Baselines")
ax.set_ylim(0.85, 1.0)
for bar, val in zip(bars, results_df["f1"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
chart_path = os.path.join(RESULTS_DIR, "model_comparison_chart_bilingual.png")
plt.savefig(chart_path, dpi=150)
plt.close()

log.info(f"Saved comparison chart -> {chart_path}")
log.info("Done.")
