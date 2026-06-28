import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve

# Dynamic folder tree matching your local Toxic_Comment_APP setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_corpus.csv")

st.set_page_config(page_title="DistilBERT Deep Learning Hub", layout="wide")

# University color accents matching your portfolio layout guidelines
st.markdown("""
    <style>
    h1 { color: #6B1F3A; font-family: 'Playfair Display', serif; font-weight: 900; }
    h3, h4 { color: #4a1228; font-family: 'Playfair Display', serif; }
    .stButton>button { background-color: #6B1F3A; color: white; border-radius: 4px; }
    .stButton>button:hover { background-color: #C9922A; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Engine 3: Custom DistilBERT Contextual Transformer")
st.markdown("---")

# =====================================================================
# 1. LIVE DATASET INGESTION
# =====================================================================
@st.cache_data
def load_clean_evaluation_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Missing dataset file! Please verify `data/balanced_corpus.csv` is present.")
        return None
    return pd.read_csv(DATA_PATH)

df_corpus = load_clean_evaluation_data()

# Isolate columns safely
text_col = "comment_text" if "comment_text" in df_corpus.columns else df_corpus.columns[0]
label_col = "is_toxic" if "is_toxic" in df_corpus.columns else df_corpus.columns[1]

# =====================================================================
# 2. DYNAMIC HUGGING FACE PIPELINE CACHING
# =====================================================================
@st.cache_resource
def load_huggingface_repo():
    """Fetches tokenizer and model layers directly from your friend's repository."""
    hf_repo = "lcokun/toxic-comment-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo)
    
    # Unified sentiment classification pipeline
    hf_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return hf_pipe

with st.spinner("Streaming DistilBERT text embedding weights from Hugging Face Hub... Please wait."):
    hf_pipeline = load_huggingface_repo()

# =====================================================================
# 3. CRASH-SAFE CALCULATION LOOP
# =====================================================================
@st.cache_data
def compile_transformer_metrics_live():
    if df_corpus is None or hf_pipeline is None:
        return None, None, None
        
    # Take a representative validation slice (50-100 samples) to ensure rapid app rendering
    sample_size = min(len(df_corpus), 100)
    eval_df = df_corpus.sample(sample_size)
    
    texts = eval_df[text_col].astype(str).tolist()
    y_true = eval_df[label_col].tolist()
    
    y_pred = []
    probabilities = []
    
    # Run predictions via deep learning inference loops
    for text in texts:
        pipe_output = hf_pipeline(text[:512])[0]
        
        raw_label = pipe_output['label']
        raw_score = pipe_output['score']
        
        if raw_label == "LABEL_1" or "toxic" in raw_label.lower() or raw_label == "1":
            toxic_prob = raw_score
            pred_class = 1
        else:
            toxic_prob = 1.0 - raw_score
            pred_class = 0
            
        probabilities.append(toxic_prob)
        y_pred.append(pred_class)
        
    return np.array(y_true), np.array(y_pred), np.array(probabilities)

# Calculate live outputs safely
y_true_real, y_pred_real, probs_real = compile_transformer_metrics_live()

# =====================================================================
# 4. CHOSEN PIPELINE METRICS INTERFACE
# =====================================================================
if y_true_real is not None:
    acc = accuracy_score(y_true_real, y_pred_real)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_real, y_pred_real, average='weighted', zero_division=0)
    
    st.write(f"### 📊 Live Performance Matrix Dashboard: **DistilBERT Deep Learning**")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Global Validation Accuracy", value=f"{acc * 100:.2f}%")
    m2.metric(label="Weighted Class Precision", value=f"{prec * 100:.2f}%")
    m3.metric(label="Weighted Class Recall", value=f"{rec * 100:.2f}%")
    m4.metric(label="Calculated F1-Score Horizon", value=f"{f1:.4f}")
    
    per_class_p, per_class_r, per_class_f, _ = precision_recall_fscore_support(y_true_real, y_pred_real, zero_division=0)
    breakdown_df = pd.DataFrame({
        "Precision (Exact Match)": per_class_p,
        "Recall (Catch Bounds)": per_class_r,
        "Calculated F1-Score": per_class_f
    }, index=["Clean / Compliant (Class 0)", "Toxic / Flagged (Class 1)"])
    st.dataframe(breakdown_df.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)

# =====================================================================
# 5. CLASS-BASED DYNAMIC WORD CLOUDS (FIXED COLORMAPS HERE)
# =====================================================================
st.markdown("---")
st.write("### ☁️ Class-Specific Document Word Clouds")
st.write("Generated live from your actual cleaned dataset rows to visualize language pattern metrics:")

col_cloud1, col_cloud2 = st.columns(2)

with col_cloud1:
    st.markdown("#### 🍏 Safe Content Word Space (Class 0)")
    clean_text_pool = " ".join(df_corpus[df_corpus[label_col] == 0][text_col].astype(str).tolist())
    if clean_text_pool.strip():
        # FIXED COLORMAP TO NATIVE MATPLOTLIB 'YlGnBu'
        wc_clean = WordCloud(width=600, height=320, background_color="#faf7f2", colormap="YlGnBu").generate(clean_text_pool)
        fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 3.2))
        ax_wc1.imshow(wc_clean, interpolation="bilinear")
        ax_wc1.axis("off")
        st.pyplot(fig_wc1)

with col_cloud2:
    st.markdown("#### 🚨 Toxic Content Word Space (Class 1)")
    toxic_text_pool = " ".join(df_corpus[df_corpus[label_col] == 1][text_col].astype(str).tolist())
    if toxic_text_pool.strip():
        # FIXED COLORMAP TO NATIVE MATPLOTLIB 'OrRd'
        wc_toxic = WordCloud(width=600, height=320, background_color="#faf7f2", colormap="OrRd").generate(toxic_text_pool)
        fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 3.2))
        ax_wc2.imshow(wc_toxic, interpolation="bilinear")
        ax_wc2.axis("off")
        st.pyplot(fig_wc2)

# =====================================================================
# 6. THREE DISTINCT VIZ CHARTS
# =====================================================================
st.markdown("---")
st.write("### 📈 Deep-Dive Diagnostic Analytics Profiles")

if y_true_real is not None:
    viz1, viz2, viz3 = st.columns(3)
    
    with viz1:
        st.markdown("#### 1. Confusion Matrix Heatmap Representation")
        cm = confusion_matrix(y_true_real, y_pred_real)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", cbar=False,
                    xticklabels=["Predicted Clean", "Predicted Toxic"],
                    yticklabels=["Actual Clean", "Actual Toxic"], ax=ax_cm)
        plt.title("DistilBERT Confusion Matrix", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.caption("Visualization 1: Error matrix mapping exact classification hits against false predictions.")

    with viz2:
        st.markdown("#### 2. Model Prediction Confidence Certainty Spread")
        fig_dist, ax_dist = plt.subplots(figsize=(4.5, 4))
        sns.histplot(probs_real, bins=15, kde=True, color="#6B1F3A", ax=ax_dist)
        ax_dist.set_xlabel("Assigned Toxicity Risk Factor Score")
        ax_dist.set_ylabel("Document Profile Densities")
        plt.title("Certainty Spread Matrix Curves", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_dist)
        st.caption("Visualization 2: Density plot mapping probability distributions assigned to text elements across the corpus.")

    with viz3:
        st.markdown("#### 3. Precision-Recall Evaluation Curve Boundaries")
        precision_array, recall_array, _ = precision_recall_curve(y_true_real, probs_real)
        
        fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4))
        ax_pr.plot(recall_array, precision_array, color="#C9922A", lw=2, label="PR Curve Boundary")
        ax_pr.set_xlabel("Recall Operational Factor")
        ax_pr.set_ylabel("Precision Operational Factor")
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlim([0.0, 1.05])
        plt.title("Precision-Recall Structural Conflict Curve", fontsize=10, pad=10)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_pr)
        st.caption("Visualization 3: Trade-off visualization checking precision preservation as retrieval recall expands.")

# =====================================================================
# 7. LIVE PLAYGROUND INPUT SANDBOX
# =====================================================================
st.markdown("---")
st.subheader("🔬 Single Instance Verification Console")
user_experiment_text = st.text_input("Type an experimental sentence structure here to scan with your Hugging Face model repository:")

if st.button("Execute Transformer Evaluation Scan") and user_experiment_text.strip() != "":
    with st.spinner("Analyzing semantic syntax vectors via transformer attention layers..."):
        single_output = hf_pipeline(user_experiment_text[:512])[0]
        
        raw_l = single_output['label']
        raw_s = single_output['score']
        
        if raw_l == "LABEL_1" or "toxic" in raw_l.lower() or raw_l == "1":
            st.error(f"🚨 **Flagged Alert:** Categorized as **TOXIC** ({raw_s*100:.2f}% risk weight assigned).")
        else:
            st.success(f"🍏 **Approved Clear:** Categorized as **CLEAN / SAFE** ({raw_s*100:.2f}% compliance weight assigned).")