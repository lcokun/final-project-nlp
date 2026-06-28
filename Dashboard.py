import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================================
# DYNAMIC PATH DETERMINATION PROFILE
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_corpus.csv")
HF_REPO = "lcokun/toxic-comment-distilbert"

st.set_page_config(page_title="Toxic Comment Moderator Hub", layout="wide")

# Custom styled branding rules (Maroon & Gold Theme profiles)
st.markdown("""
    <style>
    h1 { color: #6B1F3A; font-family: 'Playfair Display', serif; font-weight: 900; }
    h3, h4 { color: #4a1228; font-family: 'Playfair Display', serif; }
    .stButton>button { background-color: #6B1F3A; color: white; border-radius: 4px; width: 100%; }
    .stButton>button:hover { background-color: #C9922A; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Toxic Comment Detection System")
st.markdown("---")

# Cache data loading block
@st.cache_data
def get_dashboard_dataframe():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df_corpus = get_dashboard_dataframe()

if df_corpus is not None:
    # Auto-detect column headers if convention variations occur
    text_col = "comment_text" if "comment_text" in df_corpus.columns else ("preprocessed_text" if "preprocessed_text" in df_corpus.columns else df_corpus.columns[0])
    label_col = "is_toxic" if "is_toxic" in df_corpus.columns else ("toxic" if "toxic" in df_corpus.columns else df_corpus.columns[1])

    # Dynamic metrics generation metrics
    total_rows = len(df_corpus)
    class_counts = df_corpus[label_col].value_counts().to_dict()
    toxic_count = class_counts.get(1, class_counts.get('1', 0))
    clean_count = class_counts.get(0, class_counts.get('0', 0))
    balance_ratio = (toxic_count / total_rows) * 100 if total_rows > 0 else 0

    # =====================================================================
    # 1. DYNAMIC SYSTEM OVERVIEW & METRICS PROFILE
    # =====================================================================
    st.subheader("📋 System Overview & Project Dashboard")
    st.markdown("""
    Welcome to the content moderation gateway. To optimize memory consumption and execution speeds, 
    the prediction architecture has been split across dedicated feature engine workspace views.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📦 Dataset Distribution Profile")
        st.metric(label="Total Analyzed Samples", value=f"{total_rows:,} Rows", delta="Dynamic Sync Active")
        
        st.markdown(f"""
        * **Clean / Safe Comments:** {clean_count:,} rows
        * **Toxic / Flagged Comments:** {toxic_count:,} rows
        * **Balance Ratio Matrix:** ~{balance_ratio:.1f}%
        """)

    with col2:
        st.markdown("#### 🔍 Active Corpus Instance Slice")
        # Grab a randomized sample or top slice cleanly to visualize text elements
        st.dataframe(df_corpus[[text_col, label_col]].head(4), use_container_width=True)

    # =====================================================================
    # 2. XAI INTEGRATION LAB WORKSPACE (LIME & ATTENTION VISUALIZATION)
    # =====================================================================
    st.markdown("---")
    st.subheader("🔬 Component 5: Explainable AI (XAI) Diagnostic Playground")
    st.markdown("Evaluate exactly which contextual tokens or language patterns forced classifier model decisions:")

    # Core user sandbox parameters input
    xai_input = st.text_input("Type an experimental comment structure here to scan via XAI pipelines:", 
                             value="You are a stupid idiot and I will block you immediately.")

# 1. Import language detection from your preprocessing layer
from src.preprocess import detect_language

# 2. Map shorthand arrays to full readable layout blocks
LANG_MAP = {
    "en": "🇺🇸 English Language Pattern Detected",
    "ms": "🇲🇾 Malay Language Pattern Detected"
}

# 3. Dynamically compute and display the language directly on the UI
if xai_input:
    detected_code = detect_language(xai_input)
    display_name = LANG_MAP.get(detected_code, "🌐 Unknown Language Standard")
    st.info(f"**Language Analysis Matrix:** {display_name} (`{detected_code}`)")
    tab_lime, tab_attn = st.tabs(["📦 Traditional Model Interpretability (LIME)", "⚡ Transformer Attention Mappings (DistilBERT)"])

    with tab_lime:
        st.markdown("#### LIME (Local Interpretable Model-agnostic Explanations)")
        
        # Inner parameter configurations mapping dropdown triggers
        c1, c2 = st.columns(2)
        with c1:
            chosen_xai_vec = st.selectbox("Select Vectorizer Architecture:", ["bow", "tfidf"], key="xai_vec_sel")
        with c2:
            chosen_xai_mod = st.selectbox("Select Target Classifier Base:", ["logistic_regression", "naive_bayes", "svm"], key="xai_mod_sel")

        if st.button("Generate Local LIME Weights Plot"):
            model_file = f"{chosen_xai_mod}_{chosen_xai_vec}.joblib"
            vec_file = f"vectorizer_{chosen_xai_vec}.joblib"
            
            if os.path.exists(os.path.join(MODELS_PATH, model_file)) and os.path.exists(os.path.join(MODELS_PATH, vec_file)):
                with st.spinner("Calculating word contribution arrays..."):
                    clf_model = joblib.load(os.path.join(MODELS_PATH, model_file))
                    vec_transformer = joblib.load(os.path.join(MODELS_PATH, vec_file))

                    explainer = LimeTextExplainer(class_names=["Non-Toxic", "Toxic"])

                    def predict_fn(texts):
                        vecs = vec_transformer.transform(texts)
                        if hasattr(clf_model, "predict_proba"):
                            return clf_model.predict_proba(vecs)
                        else:
                            scores = clf_model.decision_function(vecs)
                            probs = 1 / (1 + np.exp(-scores))
                            return np.column_stack([1 - probs, probs])

                    exp = explainer.explain_instance(xai_input, predict_fn, num_features=8, num_samples=300, labels=[1])
                    exp_list = exp.as_list(label=1)

                # Generate the LIME horizontal bar plot
                if exp_list:
                    words = [x[0] for x in exp_list]
                    scores = [x[1] for x in exp_list]
                    colors = ["#C44E52" if s > 0 else "#4C72B0" for s in scores]

                    fig_l, ax_l = plt.subplots(figsize=(7, 3.5))
                    ax_l.barh(words[::-1], scores[::-1], color=colors[::-1])
                    ax_l.axvline(0, color="black", linewidth=0.8)
                    ax_l.set_xlabel("Contribution Weight (Positive = Toxic | Negative = Clean)")
                    plt.tight_layout()
                    st.pyplot(fig_l)
                else:
                    st.caption("No vocabulary feature mappings could be isolated.")
            else:
                st.error(f"⚠️ Model components matching `{model_file}` not found. Run your training script first.")

    with tab_attn:
        st.markdown("#### Transformer Hidden Layer Attention Matrix Mappings")
        
        @st.cache_resource
        def load_transformer_xai_assets():
            try:
                tok = AutoTokenizer.from_pretrained(HF_REPO)
                mod = AutoModelForSequenceClassification.from_pretrained(HF_REPO, output_attentions=True)
                return tok, mod
            except Exception as e:
                return None, None

        if st.button("Extract DistilBERT Attention Weights Layer"):
            with st.spinner("Downloading/Loading HuggingFace architecture and extracting final layer weights..."):
                tokenizer, bert_model = load_transformer_xai_assets()

                if bert_model is not None and tokenizer is not None:
                    bert_model.eval()
                    inputs = tokenizer(xai_input, return_tensors="pt", truncation=True, max_length=128, padding=True)

                    with torch.no_grad():
                        outputs = bert_model(**inputs, output_attentions=True)

                    # Extract last layer attention averages mapping across CLS token positions
                    last_layer_attn = outputs.attentions[-1]
                    avg_attn = last_layer_attn[0].mean(dim=0)
                    cls_attn = avg_attn[0].cpu().numpy()

                    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                    token_scores = [
                        (tok, float(score)) for tok, score in zip(tokens, cls_attn)
                        if tok not in ("[CLS]", "[SEP]", "[PAD]")
                    ]
                    token_scores.sort(key=lambda x: x[1], reverse=True)

                    if token_scores:
                        top_slice = token_scores[:12]
                        fig_a, ax_a = plt.subplots(figsize=(7, 3.5))
                        ax_a.barh([t[0] for t in top_slice][::-1], [t[1] for t in top_slice][::-1], color="#7F77DD")
                        ax_a.set_xlabel("Averaged Head Attention Matrix Scores (CLS Layer)")
                        plt.tight_layout()
                        st.pyplot(fig_a)
                    else:
                        st.caption("Input sentence context length contains too few tokens to compute context attention scores.")
                else:
                    st.error("❌ Failed connecting to HuggingFace hub to load model structures.")
else:
    st.error("❌ **Dataset Resolution Failure:** Place `balanced_corpus.csv` in your `data/` folder directory.")