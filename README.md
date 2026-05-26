# Toxic Comment Detector

A binary text classification system that detects toxic comments using classical ML models (Naive Bayes, Logistic Regression, SVM) and DistilBERT, with word-level XAI highlights served through a Streamlit app.

Built for SAIA 2163 Natural Language Processing, final project.

## Team

| Member | Role |
|--------|------|
| Amil Hakim | ML pipeline, XAI, DistilBERT fine-tuning |
| Faris Haziq | Data preprocessing, EDA |
| Irfan Johan | Streamlit app, deployment, poster |

## Dataset

The Jigsaw Toxic Comment Classification dataset must be downloaded manually from Kaggle.

1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Download `train.csv`
3. Place it in the `data/` folder

## Installation

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Running the app

```bash
uv run streamlit run app/streamlit_app.py
```
