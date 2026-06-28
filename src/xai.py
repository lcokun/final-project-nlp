import os
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Config
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE, "models")
RESULTS_DIR  = os.path.join(BASE, "results")
HF_REPO      = "lcokun/toxic-comment-distilbert"

# Load classical models
def load_classical(model_name="logistic_regression", vectorizer_name="bow"):
    model = joblib.load(f"{MODELS_DIR}/{model_name}_{vectorizer_name}.joblib")
    vec   = joblib.load(f"{MODELS_DIR}/vectorizer_{vectorizer_name}.joblib")
    return model, vec

# LIME for classical models
def explain_classical(text, model, vectorizer, num_features=10, num_samples=500):
    """
    Returns list of (word, weight) tuples sorted by absolute importance.
    Positive weight = pushes toward toxic.
    Negative weight = pushes toward clean.
    """
    explainer = LimeTextExplainer(class_names=["Non-Toxic", "Toxic"])

    def predict_fn(texts):
        vecs = vectorizer.transform(texts)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(vecs)
        else:
            # LinearSVC — use decision function + sigmoid
            scores = model.decision_function(vecs)
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])

    exp = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=[1]
    )

    return exp.as_list(label=1)

# Attention weights for DistilBERT
def explain_distilbert(text, model=None, tokenizer=None):
    """
    Returns list of (token, attention_score) tuples sorted by score descending.
    Scores averaged across all heads in the last attention layer.
    """
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO, output_attentions=True
        )
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    # take last layer, average across heads, take CLS token row
    last_layer_attn = outputs.attentions[-1]       # (1, heads, seq, seq)
    avg_attn = last_layer_attn[0].mean(dim=0)      # (seq, seq)
    cls_attn = avg_attn[0]                         # CLS attends to all tokens

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # zip tokens with attention scores, skip [CLS] and [SEP]
    token_scores = [
        (tok, float(score))
        for tok, score in zip(tokens, cls_attn)
        if tok not in ("[CLS]", "[SEP]", "[PAD]")
    ]

    # sort by score descending
    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores

# Visualisation helper
def plot_lime(lime_explanation, title="LIME Explanation", save_path=None):
    words  = [x[0] for x in lime_explanation]
    scores = [x[1] for x in lime_explanation]
    colors = ["#C44E52" if s > 0 else "#4C72B0" for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(words[::-1], scores[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution toward Toxic (positive) / Non-Toxic (negative)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()

def plot_attention(token_scores, title="Attention Weights — DistilBERT", save_path=None):
    # Take top 15 tokens
    top = token_scores[:15]
    tokens = [t[0] for t in top]
    scores = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(tokens[::-1], scores[::-1], color="#7F77DD")
    ax.set_xlabel("Attention Score (CLS token, last layer, averaged heads)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()

# Test
if __name__ == "__main__":
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_toxic = "you are a stupid idiot and i will kill you"
    test_clean = "thank you for your contribution to this article"

    print("=== LIME (Logistic Regression + BoW) ===")
    model, vec = load_classical("logistic_regression", "bow")

    for label, text in [("toxic", test_toxic), ("clean", test_clean)]:
        explanation = explain_classical(text, model, vec)
        print(f"\n[{label}] '{text[:50]}'")
        for word, score in explanation:
            print(f"  {word:20s} {score:+.4f}")
        plot_lime(
            explanation,
            title=f"LIME — {label} example",
            save_path=f"{RESULTS_DIR}/lime_{label}.png"
        )

    print("\n=== Attention Weights (DistilBERT) ===")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    model_bert = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO, output_attentions=True
    )

    for label, text in [("toxic", test_toxic), ("clean", test_clean)]:
        token_scores = explain_distilbert(text, model=model_bert, tokenizer=tokenizer)
        print(f"\n[{label}] top 10 tokens:")
        for tok, score in token_scores[:10]:
            print(f"  {tok:20s} {score:.4f}")
        plot_attention(
            token_scores,
            title=f"Attention Weights — {label} example",
            save_path=f"{RESULTS_DIR}/attention_{label}.png"
        )