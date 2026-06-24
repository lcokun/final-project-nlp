"""
Usage:
    from src.inference import load_models, predict, explain
"""

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import os
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ── Config ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models")
HF_REPO    = "lcokun/toxic-comment-distilbert"
MAX_LEN    = 128

CLASSICAL_MODELS = {
    "Logistic Regression (TF-IDF)": {
        "model": "logistic_regression_tfidf.joblib",
        "vec":   "vectorizer_tfidf.joblib",
    },
    "Logistic Regression (BoW)": {
        "model": "logistic_regression_bow.joblib",
        "vec":   "vectorizer_bow.joblib",
    },
    "SVM (TF-IDF)": {
        "model": "svm_tfidf.joblib",
        "vec":   "vectorizer_tfidf.joblib",
    },
}

# ── Model loading ──────────────────────────────────────────────────────────────
_cache = {}

def load_models():
    """
    Loads all classical models + DistilBERT into memory.
    Call once at app startup. Returns the cache dict.
    """
    global _cache
    if _cache:
        return _cache

    # classical models
    for name, paths in CLASSICAL_MODELS.items():
        model = joblib.load(os.path.join(MODELS_DIR, paths["model"]))
        vec   = joblib.load(os.path.join(MODELS_DIR, paths["vec"]))
        _cache[name] = {"model": model, "vec": vec}

    # DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO, output_attentions=True
    )
    bert_model.eval()
    _cache["DistilBERT"] = {"model": bert_model, "tokenizer": tokenizer}

    return _cache

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(text: str, model_name: str = "Logistic Regression (BoW)") -> dict:
    """
    Predicts toxicity for a given text.

    Returns:
        {
            "label":      "Toxic" or "Non-Toxic",
            "confidence": float (0-100),
            "toxic_prob": float (0-1),
            "clean_prob": float (0-1),
            "model":      str
        }
    """
    cache = load_models()

    if model_name == "DistilBERT":
        bert = cache["DistilBERT"]
        tokenizer  = bert["tokenizer"]
        bert_model = bert["model"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
        toxic_prob = float(probs[1])
        clean_prob = float(probs[0])

    else:
        entry = cache[model_name]
        model = entry["model"]
        vec   = entry["vec"]
        transformed = vec.transform([text])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(transformed)[0]
            clean_prob = float(probs[0])
            toxic_prob = float(probs[1])
        else:
            score = model.decision_function(transformed)[0]
            toxic_prob = float(1 / (1 + np.exp(-score)))
            clean_prob = 1 - toxic_prob

    label = "Toxic" if toxic_prob >= 0.5 else "Non-Toxic"
    confidence = toxic_prob * 100 if label == "Toxic" else clean_prob * 100

    return {
        "label":      label,
        "confidence": round(confidence, 2),
        "toxic_prob": round(toxic_prob, 4),
        "clean_prob": round(clean_prob, 4),
        "model":      model_name,
    }

# ── Explain ────────────────────────────────────────────────────────────────────
def explain(text: str, model_name: str = "Logistic Regression (BoW)",
            num_features: int = 10) -> list:
    """
    Returns word-level explanation for the prediction.

    For classical models: LIME (word, weight) tuples
        positive weight = pushes toward toxic
        negative weight = pushes toward clean

    For DistilBERT: attention weights (token, score) tuples
        sorted by score descending

    Returns:
        list of (word/token, score) tuples
    """
    cache = load_models()

    if model_name == "DistilBERT":
        bert      = cache["DistilBERT"]
        tokenizer = bert["tokenizer"]
        bert_model = bert["model"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True
        )
        with torch.no_grad():
            outputs = bert_model(**inputs, output_attentions=True)

        last_attn = outputs.attentions[-1]
        avg_attn  = last_attn[0].mean(dim=0)
        cls_attn  = avg_attn[0]

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_scores = [
            (tok, float(score))
            for tok, score in zip(tokens, cls_attn)
            if tok not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        token_scores.sort(key=lambda x: x[1], reverse=True)
        return token_scores[:num_features]

    else:
        entry = cache[model_name]
        model = entry["model"]
        vec   = entry["vec"]

        explainer = LimeTextExplainer(class_names=["Non-Toxic", "Toxic"])

        def predict_fn(texts):
            vecs = vec.transform(texts)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(vecs)
            else:
                scores = model.decision_function(vecs)
                probs  = 1 / (1 + np.exp(-scores))
                return np.column_stack([1 - probs, probs])

        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=500,
            labels=[1]
        )
        return [(str(word), float(score)) for word, score in exp.as_list(label=1)]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "you are a stupid idiot and i will kill you",
        "thank you for your contribution to this article",
    ]

    for model_name in [*CLASSICAL_MODELS.keys(), "DistilBERT"]:
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        for text in test_cases:
            result = predict(text, model_name)
            explanation = explain(text, model_name)
            print(f"\nText: {text[:50]}")
            print(f"  → {result['label']} ({result['confidence']:.1f}%)")
            print(f"  Top words: {explanation[:3]}")