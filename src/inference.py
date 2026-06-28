import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import os
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

from src.preprocess import detect_language, preprocess

# ── Config ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models")
HF_REPO           = "lcokun/toxic-comment-distilbert"
HF_REPO_BILINGUAL = "lcokun/toxic-comment-xlm-roberta-bilingual"
LOCAL_BILINGUAL   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xlm_roberta_bilingual")
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

BILINGUAL_MODELS = {
    "Logistic Regression (TF-IDF)": {
        "model": "logistic_regression_tfidf_bilingual.joblib",
        "vec":   "vectorizer_tfidf_bilingual.joblib",
    },
    "Logistic Regression (BoW)": {
        "model": "logistic_regression_bow_bilingual.joblib",
        "vec":   "vectorizer_bow_bilingual.joblib",
    },
    "SVM (TF-IDF)": {
        "model": "svm_tfidf_bilingual.joblib",
        "vec":   "vectorizer_tfidf_bilingual.joblib",
    },
    "Naive Bayes (TF-IDF)": {
        "model": "naive_bayes_tfidf_bilingual.joblib",
        "vec":   "vectorizer_tfidf_bilingual.joblib",
    },
    "Naive Bayes (BoW)": {
        "model": "naive_bayes_bow_bilingual.joblib",
        "vec":   "vectorizer_bow_bilingual.joblib",
    },
    "SVM (BoW)": {
        "model": "svm_bow_bilingual.joblib",
        "vec":   "vectorizer_bow_bilingual.joblib",
    },
}

# ── Model loading ──────────────────────────────────────────────────────────────
_cache = {}

def load_models():
    global _cache
    if _cache:
        return _cache

    for name, paths in CLASSICAL_MODELS.items():
        model = joblib.load(os.path.join(MODELS_DIR, paths["model"]))
        vec   = joblib.load(os.path.join(MODELS_DIR, paths["vec"]))
        _cache[f"en::{name}"] = {"model": model, "vec": vec}

    for name, paths in BILINGUAL_MODELS.items():
        model = joblib.load(os.path.join(MODELS_DIR, paths["model"]))
        vec   = joblib.load(os.path.join(MODELS_DIR, paths["vec"]))
        _cache[f"bi::{name}"] = {"model": model, "vec": vec}

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO, output_attentions=True
    )
    bert_model.eval()
    _cache["DistilBERT"] = {"model": bert_model, "tokenizer": tokenizer}

    bi_src = LOCAL_BILINGUAL if os.path.isdir(LOCAL_BILINGUAL) else HF_REPO_BILINGUAL
    bi_tokenizer = AutoTokenizer.from_pretrained(bi_src)
    bi_model = AutoModelForSequenceClassification.from_pretrained(
        bi_src, output_attentions=True
    )
    bi_model.eval()
    _cache["XLM-RoBERTa (Bilingual)"] = {"model": bi_model, "tokenizer": bi_tokenizer}

    return _cache


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(text: str, model_name: str, lang: str | None = None) -> dict:
    """
    Predicts toxicity for a given text.

    model_name: display name from CLASSICAL_MODELS / BILINGUAL_MODELS, or
                "DistilBERT" / "DistilBERT Multilingual"
    lang: 'en' or 'ms'. Auto-detected if None.

    Returns:
        {
            "label":      "Toxic" or "Non-Toxic",
            "confidence": float (0-100),
            "toxic_prob": float (0-1),
            "clean_prob": float (0-1),
            "model":      str,
            "lang":       str,
        }
    """
    cache = load_models()

    if lang is None:
        lang = detect_language(text)

    if model_name in ("DistilBERT", "XLM-RoBERTa (Bilingual)"):
        key = model_name
        bert      = cache[key]
        tokenizer = bert["tokenizer"]
        bert_model = bert["model"]

        # XLM-RoBERTa Malay was trained on raw text — skip preprocessing for MS.
        # English side was still preprocessed, so preprocess EN as before.
        if model_name == "XLM-RoBERTa (Bilingual)" and lang == "ms":
            processed = text
        else:
            processed = preprocess(text, lang)

        inputs = tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
        toxic_prob = float(probs[1])
        clean_prob = float(probs[0])

    else:
        is_bilingual = (lang == "ms")
        cache_key = f"{'bi' if is_bilingual else 'en'}::{model_name}"
        entry = cache[cache_key]
        model = entry["model"]
        vec   = entry["vec"]

        processed = preprocess(text, lang)
        transformed = vec.transform([processed])

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
        "lang":       lang,
    }


# ── Explain ────────────────────────────────────────────────────────────────────
def explain(text: str, model_name: str, lang: str | None = None,
            num_features: int = 10) -> list:
    """
    Returns word-level explanation for the prediction.
    lang: 'en' or 'ms'. Auto-detected if None.
    """
    cache = load_models()

    if lang is None:
        lang = detect_language(text)

    if model_name == "XLM-RoBERTa (Bilingual)" and lang == "ms":
        processed = text
    else:
        processed = preprocess(text, lang)

    if model_name in ("DistilBERT", "XLM-RoBERTa (Bilingual)"):
        key = model_name
        bert       = cache[key]
        tokenizer  = bert["tokenizer"]
        bert_model = bert["model"]

        inputs = tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
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
        is_bilingual = (lang == "ms")
        cache_key = f"{'bi' if is_bilingual else 'en'}::{model_name}"
        entry = cache[cache_key]
        model = entry["model"]
        vec   = entry["vec"]

        explainer = LimeTextExplainer(class_names=["Non-Toxic", "Toxic"])

        def predict_fn(texts):
            preprocessed = [preprocess(t, lang) for t in texts]
            vecs = vec.transform(preprocessed)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(vecs)
            else:
                scores = model.decision_function(vecs)
                probs  = 1 / (1 + np.exp(-scores))
                return np.column_stack([1 - probs, probs])

        exp = explainer.explain_instance(
            processed,
            predict_fn,
            num_features=num_features,
            num_samples=500,
            labels=[1],
        )
        return [(str(word), float(score)) for word, score in exp.as_list(label=1)]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("you are a stupid idiot and i will kill you", "en"),
        ("thank you for your contribution to this article", "en"),
        ("bodoh sial kau ni, pergi mampus", "ms"),
        ("terima kasih atas sokongan anda, sangat menghargai", "ms"),
    ]

    for model_name in [*CLASSICAL_MODELS.keys(), "DistilBERT", "XLM-RoBERTa (Bilingual)"]:
        print(f"\n{'='*55}")
        print(f"Model: {model_name}")
        print(f"{'='*55}")
        for text, lang in test_cases:
            result = predict(text, model_name, lang=lang)
            print(f"  [{lang}] {text[:45]:<45} -> {result['label']} ({result['confidence']:.1f}%)")
