---
title: Toxic Comment Detection System
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.0
app_file: Dashboard.py
python_version: "3.11"
pinned: false
license: mit
---

# Toxic Comment Detector

A bilingual (English + Malay) toxic comment detection system using classical ML and transformer models, with word-level explainability via LIME and attention weights, served through a Streamlit app.

Built for SAIA 2163 Natural Language Processing, final project.

**Live demo:** https://huggingface.co/spaces/lcokun/nlp-showcase

## Team

| Member | Role |
|--------|------|
| Amil Hakim | ML pipeline, bilingual models, XAI, inference module |
| Faris Haziq | Dataset preprocessing, bilingual corpus, EDA |
| Irfan Johan | Streamlit app, deployment, poster |

## Models

### English

| Model | Vectorizer | F1 |
|-------|------------|----|
| DistilBERT (fine-tuned) | | 0.9563 |
| Logistic Regression | BoW | 0.9036 |
| SVM | TF-IDF | 0.8995 |
| Logistic Regression | TF-IDF | 0.8980 |
| SVM | BoW | 0.8892 |
| Naive Bayes | TF-IDF | 0.8797 |
| Naive Bayes | BoW | 0.8775 |

### Bilingual (English + Malay)

| Model | Vectorizer | F1 |
|-------|------------|----|
| XLM-RoBERTa (fine-tuned) | | 0.90+ |
| Logistic Regression | BoW | 0.8936 |
| SVM | BoW | 0.8917 |
| SVM | TF-IDF | 0.8909 |
| Logistic Regression | TF-IDF | 0.8895 |
| Naive Bayes | BoW | 0.8589 |
| Naive Bayes | TF-IDF | 0.8549 |

Pretrained models on HuggingFace:
- English DistilBERT: https://huggingface.co/lcokun/toxic-comment-distilbert
- Bilingual XLM-RoBERTa: https://huggingface.co/lcokun/toxic-comment-xlm-roberta-bilingual

## Dataset

**English:** Jigsaw Toxic Comment Classification Challenge (Kaggle)

1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Download `train.csv`
3. Place it in `data/`

**Bilingual:** A Bilingual Malay-English Social Media Dataset for Binary Hate Speech Detection (13,376 Malay rows from HateM, Toxicity-Small, Snapshot-Twitter-2022, Supervised-Twitter).

Preprocessed balanced corpus included in the repo:
- `data/balanced_corpus.csv` — English only, 30,568 rows (50/50 balanced)

## Installation

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/lcokun/final-project-nlp
cd final-project-nlp
uv sync
```

## Running the app locally

```bash
uv run streamlit run Dashboard.py
```

## Repository structure

```
final-project-nlp/
├── Dashboard.py                    # Streamlit app entry point
├── pages/
│   ├── Bert.py                     # DistilBERT / XLM-RoBERTa evaluation page
│   ├── BoW.py                      # Bag-of-Words model evaluation page
│   └── TF-IDF.py                   # TF-IDF model evaluation page
├── requirements.txt
├── pyproject.toml
├── data/
│   └── balanced_corpus.csv         # English balanced corpus
├── models/                         # Classical model .joblib files (English + bilingual)
├── notebooks/
│   ├── Text_preprocessing_checkpoint1_group_amil.ipynb
│   ├── Text_preprocessing_checkpoint1_MalayEng_ver.ipynb
│   └── SAIA2163_FinalProject_Colab.ipynb
├── results/                        # Confusion matrices, comparison charts
└── src/
    ├── preprocess.py               # Inference-time preprocessing (EN + MS)
    ├── inference.py                # predict() and explain() API
    ├── xai.py                      # LIME and attention weight helpers
    ├── train_classical.py          # Train English classical models
    ├── train_classical_bilingual.py
    ├── train_distillbert.py
    ├── train_distillbert_bilingual.py
    ├── train_xlm_roberta_bilingual.py
    └── rebuild_bilingual_corpus.py
```

## Inference API

```python
from src.inference import predict, explain

# language auto-detected
result = predict("bodoh sial kau ni", "XLM-RoBERTa (Bilingual)")
# {"label": "Toxic", "confidence": 99.7, "toxic_prob": 0.997, "lang": "ms", ...}

explanation = explain("you are an idiot", "Logistic Regression (BoW)")
# [(word, weight), ...] — positive weight = pushes toward toxic
```

## Results

Full evaluation results and visualizations are in `results/`. The Colab notebook (`notebooks/SAIA2163_FinalProject_Colab.ipynb`) demonstrates the full pipeline end-to-end without retraining.
