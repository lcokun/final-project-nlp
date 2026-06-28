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

# Import language detection from your local preprocessing layer
try:
    from src.preprocess import detect_language
except ImportError:
    def detect_language(text):
        text_lower = text.lower()
        malay_words = ["saya", "kamu", "anda", "tidak", "tak", "bukan", "adalah", "yang", "dan", "di", "ke"]
        if any(w in text_lower.split() for w in malay_words):
            return "ms"
        return "en"

# =====================================================================
# DYNAMIC PATH DETERMINATION PROFILE
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_corpus.csv")

# Model Repositories
HF_REPO_EN = "lcokun/toxic-comment-distilbert"
HF_REPO_BI = "lcokun/toxic-comment-xlm-roberta-bilingual"

LANG_MAP = {
    "en": "🇺🇸 English Language Pattern Detected",
    "ms": "🇲🇾 Malay Language Pattern Detected"
}

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
    # Auto-detect column headers
    text_col = "comment_text" if "comment_text" in df_corpus.columns else ("preprocessed_text" if "preprocessed_text" in df_corpus.columns else df_corpus.columns[0])
    label_col = "is_toxic" if "is_toxic" in df_corpus.columns else ("toxic" if "toxic" in df_corpus.columns else df_corpus.columns[1])

    # Dynamic metrics generation
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
    Welcome to the content moderation gateway. The prediction architecture has been split 
    across dedicated linguistic and architectural feature engine workspaces.
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
        st.markdown("#### 🔍 Corpus Datasets")
        
        # Created tabs to separate the random sampler from the text search features cleanly
        corpus_tab1, corpus_tab2 = st.tabs(["🎲 Randomized Instance Slice", "🔍 Search Dataset Engine"])
        
        with corpus_tab1:
            random_slice = df_corpus[[text_col, label_col]].sample(4)
            st.dataframe(random_slice, use_container_width=True)
            st.caption("💡 *Tip: Click 'Refresh' or interact with any widget to generate a brand new random sequence stream.*")
            
        with corpus_tab2:
            search_query = st.text_input("Enter search keywords or phrase patterns:", placeholder="e.g., stupid, love, benci...", key="corpus_txt_search")
            if search_query.strip():
                matched_df = df_corpus[df_corpus[text_col].str.contains(search_query, case=False, na=False)]
                total_matches = len(matched_df)
                
                st.write(f"🎯 **Matches Found:** {total_matches:,} rows match your query.")
                if total_matches > 0:
                    st.dataframe(matched_df[[text_col, label_col]].head(10), use_container_width=True)
                else:
                    st.warning("No matches discovered within the source text columns.")
            else:
                st.caption("Type a target keyword above to automatically query your balanced corpus matrix.")

    st.markdown("---")

    # =====================================================================
    # SIDEBAR INTERACTIVE TIERS CONTROL & PROJECT TEAM MEMBERS
    # =====================================================================
    st.sidebar.header("🛠️ Language Scope Control")
    model_tier = st.sidebar.radio(
        "Choose Target Language Scope:",
        ["🏛️ English-Only Models", "🌐 Bilingual Models (EN/MS)"]
    )

    st.sidebar.markdown("---")
    st.sidebar.header("👥 Project Team Members")
    
    # Rendered team profile elements layout blocks
    st.sidebar.markdown("""
    🟢 **Amil Hakim** *ML pipeline, bilingual models, XAI, inference module* \n
    🟢 **Faris Haziq** *Dataset preprocessing, bilingual corpus, EDA* \n
    🟢 **Irfan Johan** *Streamlit app, deployment, poster*
    """)
    st.sidebar.markdown("---")

    # Global Sandbox Playground Entry
    st.subheader("🔬 Explainable AI (XAI) Diagnostic")
    st.markdown("Evaluate exactly which contextual tokens or language patterns forced classifier model decisions:")

    xai_input = st.text_input(
        "Type an experimental comment structure here to scan via active pipelines:", 
        value="You are a stupid idiot and I will block you immediately.",
        key="playground_main_input"
    )

    # Set file suffix dynamically based on sidebar scope selection
    suffix = "_bilingual" if "Bilingual" in model_tier else ""

    # Shared helper function to cache transformer models safely
    @st.cache_resource
    def load_transformer_assets(repo_path):
        try:
            tok = AutoTokenizer.from_pretrained(repo_path)
            mod = AutoModelForSequenceClassification.from_pretrained(repo_path, output_attentions=True)
            return tok, mod
        except Exception as e:
            return None, None

    # =====================================================================
    # TIER A: ENGLISH-ONLY MODELS VIEW (Classical + DistilBERT)
    # =====================================================================
    if model_tier == "🏛️ English-Only Models":
        st.markdown("### 🏛️ English-Only Language Gateway")
        
        if xai_input:
            en_mode = st.radio(
                "Select Framework Strategy:",
                ["Classical ML", "Transformer (DistilBERT)"]
            )
            
            # --- Sub-Option 1: English Classical Models ---
            if "Classical ML" in en_mode:
                st.markdown("#### Evaluating Classical English Pipeline Weights")
                c1, c2 = st.columns(2)
                with c1:
                    chosen_xai_vec = st.selectbox("Select Vectorizer Architecture:", ["bow", "tfidf"], key="en_vec_sel")
                with c2:
                    chosen_xai_mod = st.selectbox("Select Target Classifier Base:", ["logistic_regression", "naive_bayes", "svm"], key="en_mod_sel")

                if st.button("Generate English LIME Weights Plot", key="en_classical_btn"):
                    model_file = f"{chosen_xai_mod}_{chosen_xai_vec}{suffix}.joblib"
                    vec_file = f"vectorizer_{chosen_xai_vec}{suffix}.joblib"
                    
                    model_filepath = os.path.join(MODELS_PATH, model_file)
                    vec_filepath = os.path.join(MODELS_PATH, vec_file)

                    if os.path.exists(model_filepath) and os.path.exists(vec_filepath):
                        with st.spinner("Calculating word contribution arrays..."):
                            clf_model = joblib.load(model_filepath)
                            vec_transformer = joblib.load(vec_filepath)

                            # Version Mismatch Attribute Patch
                            if not hasattr(clf_model, 'multi_class'):
                                clf_model.multi_class = 'auto'

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
                        st.error(f"⚠️ Missing file tracking: Expected `{model_file}` and `{vec_file}` inside your `/models` directory.")
            
            # --- Sub-Option 2: DistilBERT ---
            else:
                st.markdown("#### DistilBERT Hidden Layer Attention Context Profile")
                
                if st.button("Extract DistilBERT Attention Weights Layer", key="en_trans_btn"):
                    with st.spinner("Loading DistilBERT architecture and extracting final layer weights..."):
                        tokenizer, bert_model = load_transformer_assets(HF_REPO_EN)

                        if bert_model is not None and tokenizer is not None:
                            bert_model.eval()
                            inputs = tokenizer(xai_input, return_tensors="pt", truncation=True, max_length=128, padding=True)

                            with torch.no_grad():
                                outputs = bert_model(**inputs, output_attentions=True)

                            last_layer_attn = outputs.attentions[-1]
                            avg_attn = last_layer_attn[0].mean(dim=0)
                            cls_attn = avg_attn[0].cpu().numpy()

                            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                            token_scores = [
                                (tok, float(score)) for tok, score in zip(tokens, cls_attn)
                                if tok not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>")
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
                                st.caption("Input sentence context contains too few valid tokens to visualize attention.")
                        else:
                            st.error(f"❌ Failed connecting to HuggingFace hub to load model: `{HF_REPO_EN}`.")

    # =====================================================================
    # TIER B: BILINGUAL MODELS VIEW (Classical + XLM-RoBERTa)
    # =====================================================================
    elif model_tier == "🌐 Bilingual Models (EN/MS)":
        st.markdown("### 🌐 Bilingual Language (Malay & English)")
        
        if xai_input:
            detected_code = detect_language(xai_input)
            display_name = LANG_MAP.get(detected_code, "🌐 Unknown Language Standard")
        
            
            bilingual_mode = st.radio(
                "Select Bilingual Framework Strategy:",
                ["Classical ML", "Transformer (XLM-RoBERTa Bilingual)"]
            )
            
            # --- Sub-Option 1: Bilingual Classical Models ---
            if "Classical ML" in bilingual_mode:
                st.markdown("#### Evaluating Classical Bilingual Pipeline Weights")
                c1, c2 = st.columns(2)
                with c1:
                    chosen_xai_vec = st.selectbox("Select Vectorizer Architecture:", ["bow", "tfidf"], key="bi_vec_sel")
                with c2:
                    chosen_xai_mod = st.selectbox("Select Target Classifier Base:", ["logistic_regression", "naive_bayes", "svm"], key="bi_mod_sel")

                if st.button("Generate Bilingual LIME Weights Plot", key="bi_classical_btn"):
                    model_file = f"{chosen_xai_mod}_{chosen_xai_vec}{suffix}.joblib"
                    vec_file = f"vectorizer_{chosen_xai_vec}{suffix}.joblib"
                    
                    model_filepath = os.path.join(MODELS_PATH, model_file)
                    vec_filepath = os.path.join(MODELS_PATH, vec_file)

                    if os.path.exists(model_filepath) and os.path.exists(vec_filepath):
                        with st.spinner("Calculating bilingual word token interactions..."):
                            clf_model = joblib.load(model_filepath)
                            vec_transformer = joblib.load(vec_filepath)

                            # Version Mismatch Attribute Patch
                            if not hasattr(clf_model, 'multi_class'):
                                clf_model.multi_class = 'auto'

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
                        st.error(f"⚠️ Missing bilingual file: Expected `{model_file}` and `{vec_file}` inside your `/models` directory.")
            
            # --- Sub-Option 2: XLM-RoBERTa Bilingual ---
            else:
                st.markdown("#### XLM-RoBERTa Hidden Layer Attention Extract")
                
                if st.button("Extract XLM-RoBERTa Attention Weights Layer", key="bi_trans_btn"):
                    with st.spinner("Loading XLM-RoBERTa Bilingual architecture and extracting final layer weights..."):
                        tokenizer, bert_model = load_transformer_assets(HF_REPO_BI)

                        if bert_model is not None and tokenizer is not None:
                            bert_model.eval()
                            inputs = tokenizer(xai_input, return_tensors="pt", truncation=True, max_length=128, padding=True)

                            with torch.no_grad():
                                outputs = bert_model(**inputs, output_attentions=True)

                            last_layer_attn = outputs.attentions[-1]
                            avg_attn = last_layer_attn[0].mean(dim=0)
                            cls_attn = avg_attn[0].cpu().numpy()

                            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                            token_scores = [
                                (tok, float(score)) for tok, score in zip(tokens, cls_attn)
                                if tok not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>")
                            ]
                            token_scores.sort(key=lambda x: x[1], reverse=True)

                            if token_scores:
                                top_slice = token_scores[:12]
                                fig_a, ax_a = plt.subplots(figsize=(7, 3.5))
                                ax_a.barh([t[0] for t in top_slice][::-1], [t[1] for t in top_slice][::-1], color="#E67E22")
                                ax_a.set_xlabel("Averaged Head Attention Matrix Scores (Classification Token Layer)")
                                plt.tight_layout()
                                st.pyplot(fig_a)
                            else:
                                st.caption("Input context sequence contains too few valid items to render attention scores.")
                        else:
                            st.error(f"❌ Failed connecting to HuggingFace hub to load model: `{HF_REPO_BI}`.")
else:
    st.error(f"❌ **Dataset Resolution Failure:** Could not locate files at path: `{DATA_PATH}`.")