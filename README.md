# Toxic Comment Detector

A binary text classification system that detects toxic comments using classical ML models (Logistic Regression, SVM) and DistilBERT, with word-level XAI highlights via LIME and attention weights, served through a Streamlit app.

Built for SAIA 2163 Natural Language Processing, final project.

## Team

| Member | Role |
|--------|------|
| Amil Hakim | ML pipeline, XAI, DistilBERT fine-tuning |
| Faris Haziq | Data preprocessing, EDA |
| Irfan Johan | Streamlit app, deployment, poster |

## Models

Three classical models and one transformer model are available in the app:

| Model | F1 Score |
|-------|----------|
| DistilBERT | 0.9563 |
| Logistic Regression (BoW) | 0.9036 |
| SVM (TF-IDF) | 0.8995 |
| Logistic Regression (TF-IDF) | 0.8980 |

Pretrained DistilBERT model: https://huggingface.co/lcokun/toxic-comment-distilbert

## Dataset

The Jigsaw Toxic Comment Classification dataset must be downloaded manually from Kaggle.

1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Download `train.csv`
3. Place it in the `data/` folder

A preprocessed balanced corpus (`balanced_corpus.csv`) is included in `data/` for convenience.

## Installation

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/lcokun/final-project-nlp
cd final-project-nlp
uv sync
```

## Running the app

```bash
uv run --no-project streamlit run app.py
```

## Repository structure

```
final-project-nlp/
├── app.py              # Streamlit app
├── requirements.txt
├── pyproject.toml
├── data/
│   └── balanced_corpus.csv
├── models/             # Pretrained classical models (.joblib)
├── notebooks/          # Jupyter notebook and Colab submission
├── results/            # Evaluation plots and confusion matrices
└── src/
    ├── train_classical.py
    ├── train_distillbert.py
    ├── xai.py
    └── inference.py
```

## Results

Full evaluation results and visualizations are in `results/`. The Colab notebook (`notebooks/SAIA2163_FinalProject_Colab.ipynb`) demonstrates the full pipeline end-to-end without retraining.
