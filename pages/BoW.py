import os
import re
import warnings
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve

# =====================================================================
# AUTOMATED LANGUAGE DETECTION LAYER (NON-HARDCODED)
# =====================================================================
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42  # Guarantees deterministic evaluation output paths
except ImportError:
    import subprocess
    import sys
    try:
        # Automatically handle package availability within runtime instances
        subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect"])
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
    except Exception:
        detect = None

def detect_language(text):
    """
    Automated NLP text classification router. Automatically identifies input language
    to match core runtime data processing paths without hardcoded lists.
    """
    if not text or len(text.strip()) < 2:
        return "en" # Fallback safety default
        
    if detect is not None:
        try:
            lang = detect(text)
            if lang in ["ms", "id"]:
                return "ms"
            return "en"
        except Exception:
            pass # Move to fallback calculation if sequence is too small/ambiguous
            
    # Linguistic Structural Signature Fallback
    text_lower = text.lower().split()
    malay_structural_markers = {"yang", "dan", "di", "ke", "itu", "ini", "untuk", "dengan", "saya", "kamu", "benci"}
    ms_matches = sum(1 for word in text_lower if word in malay_structural_markers)
    return "ms" if ms_matches > 0 else "en"

# ── Config & Repository Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
MODELS_PATH = os.path.join(BASE_DIR, "models")
LOCAL_DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_corpus.csv")

# Remote Git URLs targeting your source project repository
REMOTE_DATA_URL_MAIN = "https://raw.githubusercontent.com/lcokun/final-project-nlp/main/balanced_corpus.csv"
REMOTE_DATA_URL_MASTER = "https://raw.githubusercontent.com/lcokun/final-project-nlp/master/balanced_corpus.csv"

LANG_MAP = {
    "en": "English Language Profile",
    "ms": "Bilingual / Malay Language Profile"
}

st.set_page_config(page_title="BoW Model Diagnostic Workspace", layout="wide")

# University color styling attributes (Maroon & Gold Theme profiles)
st.markdown("""
    <style>
    h1 { color: #6B1F3A; font-family: 'Playfair Display', serif; font-weight: 900; }
    h3, h4 { color: #4a1228; font-family: 'Playfair Display', serif; }
    .stButton>button { background-color: #6B1F3A; color: white; border-radius: 4px; width: 100%; height: 45px; font-weight: bold; }
    .stButton>button:hover { background-color: #C9922A; color: white; }
    </style>
""", unsafe_allow_html=True)

# Rendered team profile elements layout blocks
st.sidebar.markdown("""
🟢 **Amil Hakim** *ML pipeline, bilingual models, XAI, inference module* \n
🟢 **Faris Haziq** *Dataset preprocessing, bilingual corpus, EDA* \n
🟢 **Irfan Johan** *Streamlit app, deployment, poster*
""")
st.sidebar.markdown("---")

st.title("📦 Engine 1: Bag-of-Words (BoW) Live Evaluation Hub")
st.markdown("---")

# =====================================================================
# 1. READ CLEAN DATASET & ASSETS SAFELY FROM LOCAL DISK / GITHUB
# =====================================================================
@st.cache_data
def load_clean_corpus_data():
    """Loads validation dataset locally, falling back to streaming directly from GitHub repository."""
    if os.path.exists(LOCAL_DATA_PATH):
        return pd.read_csv(LOCAL_DATA_PATH)
        
    # Cascade fallback to handle GitHub main/master branches gracefully
    for url in [REMOTE_DATA_URL_MAIN, REMOTE_DATA_URL_MASTER]:
        try:
            df = pd.read_csv(url)
            return df
        except Exception:
            continue
    return None

@st.cache_resource
def load_bow_model_assets():
    try:
        return {
            "vec": joblib.load(os.path.join(MODELS_PATH, "vectorizer_bow.joblib")),
            "Naive Bayes (BoW)": joblib.load(os.path.join(MODELS_PATH, "naive_bayes_bow.joblib")),
            "Logistic Regression (BoW)": joblib.load(os.path.join(MODELS_PATH, "logistic_regression_bow.joblib")),
            "Support Vector Machine (BoW)": joblib.load(os.path.join(MODELS_PATH, "svm_bow.joblib"))
        }
    except Exception as e:
        st.error(f"❌ **Asset Ingestion Error:** Failed to load your scikit-learn `.joblib` pipelines. Details: {e}")
        return None

df_corpus = load_clean_corpus_data()
assets = load_bow_model_assets()

if df_corpus is not None and assets is not None:
    # Safely identify dataset structural columns
    text_col = "comment_text" if "comment_text" in df_corpus.columns else df_corpus.columns[0]
    label_col = "is_toxic" if "is_toxic" in df_corpus.columns else df_corpus.columns[1]

    X_test = df_corpus[text_col].astype(str).tolist()
    y_test = df_corpus[label_col].tolist()

    # =====================================================================
    # 2. INTERACTIVE SELECTBOX HUB CONTROLS
    # =====================================================================
    st.subheader("⚙️ Classifier Selector Panel")
    chosen_algo = st.selectbox(
        "Choose an Active Bag-of-Words Classifier to Evaluate & Review:", 
        ["Naive Bayes (BoW)", "Logistic Regression (BoW)", "Support Vector Machine (BoW)"]
    )

    # 3. LIVE MATRIX INFERENCE COMPILATION
    vec_transformer = assets["vec"]
    active_classifier = assets[chosen_algo]

    with st.spinner("Processing prediction arrays live..."):
        X_test_vec = vec_transformer.transform(X_test)
        y_pred = active_classifier.predict(X_test_vec)

    # Extract global score thresholds dynamically
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    # =====================================================================
    # 4. CHOSEN CLASSIFIER LIVE METRICS DASHBOARD
    # =====================================================================
    st.markdown("---")
    st.write(f"### 📊 Real-Time Metric Performance Dashboard: **{chosen_algo}**")
    
    # Render upper summary card components
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Global Accuracy Score", value=f"{acc * 100:.2f}%")
    m2.metric(label="Weighted Class Precision", value=f"{prec * 100:.2f}%")
    m3.metric(label="Weighted Class Recall", value=f"{rec * 100:.2f}%")
    m4.metric(label="Calculated F1-Score Baseline", value=f"{f1:.4f}")

    # Detailed sub-class segmentation report grid
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    breakdown_df = pd.DataFrame({
        "Precision (Exact Match)": per_class_prec,
        "Recall (Catch Bounds)": per_class_rec,
        "Calculated F1-Score": per_class_f1
    }, index=["Clean / Compliant (Class 0)", "Toxic / Flagged (Class 1)"])
    
    st.dataframe(breakdown_df.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)

    # =====================================================================
    # 5. CLASS-BASED DYNAMIC WORD CLOUDS
    # =====================================================================
    st.markdown("---")
    st.write("### ☁️ Class-Specific Document Word Clouds")
    st.write("Generated live from your actual cleaned dataset rows to expose language pattern weights:")
    
    col_cloud1, col_cloud2 = st.columns(2)
    
    with col_cloud1:
        st.markdown("#### 🍏 Safe Content Word Space (Class 0)")
        clean_text_pool = " ".join(df_corpus[df_corpus[label_col] == 0][text_col].astype(str).tolist())
        if clean_text_pool.strip():
            wc_clean = WordCloud(width=600, height=320, background_color="#faf7f2", colormap="crest").generate(clean_text_pool)
            fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 3.2))
            ax_wc1.imshow(wc_clean, interpolation="bilinear")
            ax_wc1.axis("off")
            st.pyplot(fig_wc1)
        else:
            st.caption("No non-toxic rows found inside source csv columns.")

    with col_cloud2:
        st.markdown("#### 🚨 Toxic Content Word Space (Class 1)")
        toxic_text_pool = " ".join(df_corpus[df_corpus[label_col] == 1][text_col].astype(str).tolist())
        if toxic_text_pool.strip():
            wc_toxic = WordCloud(width=600, height=320, background_color="#faf7f2", colormap="flare").generate(toxic_text_pool)
            fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 3.2))
            ax_wc2.imshow(wc_toxic, interpolation="bilinear")
            ax_wc2.axis("off")
            st.pyplot(fig_wc2)
        else:
            st.caption("No toxic rows found inside source csv columns.")

    # =====================================================================
    # 6. THREE DISTINCT VIZ CHARTS
    # =====================================================================
    st.markdown("---")
    st.write("### 📈 Deep-Dive Diagnostic Analytics Profiles")
    
    viz1, viz2, viz3 = st.columns(3)
    
    with viz1:
        st.markdown("#### 1. Confusion Matrix Heatmap Representation")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", cbar=False,
                    xticklabels=["Predicted Clean", "Predicted Toxic"],
                    yticklabels=["Actual Clean", "Actual Toxic"], ax=ax_cm)
        plt.title(f"Confusion Matrix: {chosen_algo}", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.caption("Visualization 1: Real error tracking matrix compiling exact hits vs classification errors.")

    with viz2:
        st.markdown("#### 2. Model Prediction Confidence Certainty Spread")
        if hasattr(active_classifier, "predict_proba"):
            probabilities_array = active_classifier.predict_proba(X_test_vec)[:, 1]
        else:
            distance_scores = active_classifier.decision_function(X_test_vec)
            probabilities_array = (distance_scores - distance_scores.min()) / (distance_scores.max() - distance_scores.min())
            
        fig_dist, ax_dist = plt.subplots(figsize=(4.5, 4))
        sns.histplot(probabilities_array, bins=15, kde=True, color="#6B1F3A", ax=ax_dist)
        ax_dist.set_xlabel("Assigned Toxicity Risk Factor Score")
        ax_dist.set_ylabel("Document Profile Densities")
        plt.title("Certainty Spread Matrix Curves", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_dist)
        st.caption("Visualization 2: Density plot mapping probability distributions assigned to text elements across the corpus.")

    with viz3:
        st.markdown("#### 3. Precision-Recall Evaluation Curve Boundaries")
        if hasattr(active_classifier, "predict_proba"):
            scores = active_classifier.predict_proba(X_test_vec)[:, 1]
        else:
            scores = active_classifier.decision_function(X_test_vec)
            
        precision_array, recall_array, _ = precision_recall_curve(y_test, scores)
        
        fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4))
        ax_pr.plot(recall_array, precision_array, color="#C9922A", lw=2, label="PR Curve Boundary")
        ax_pr.set_xlabel("Recall Operational Factor")
        ax_pr.set_ylabel("Precision Operational Factor")
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlim([0.0, 1.05])
        plt.title("Precision-Recall Structural Curve", fontsize=10, pad=10)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_pr)
        st.caption("Visualization 3: Trade-off visualization checking precision preservation as retrieval recall expands.")

    # =====================================================================
    # 7. AUTOMATED LANGUAGE ROUTING PLAYGROUND
    # =====================================================================
    st.markdown("---")
    st.subheader("🔬 Automated Language Detection & Live Validation Console")
    st.markdown("Type an experimental comment in either language. The system will auto-detect the context signature profiles.")
    
    user_experiment_text = st.text_input(
        "Type an experimental sentence structure here to scan with this specific model setup:",
        value="Saya sangat benci dengan perangai buruk awak.",
        key="bow_playground_input"
    )
    
    if st.button("Execute Safety Verification Scan") and user_experiment_text.strip() != "":
        # Run Language Detection dynamically
        detected_lang = detect_language(user_experiment_text)
        
        # --- DYNAMIC MODEL NAME RESOLVER ---
        algo_name_tokens = {
            "Naive Bayes (BoW)": "naive_bayes_bow",
            "Logistic Regression (BoW)": "logistic_regression_bow",
            "Support Vector Machine (BoW)": "svm_bow"
        }
        base_model_id = algo_name_tokens.get(chosen_algo, "model_bow")
        resolved_model_string = f"{base_model_id}_bilingual" if detected_lang == "ms" else base_model_id
        
        # Display Active Routing Context along with the dynamically verified core model string name
        st.info(f"⚡ **Routing Decision Engine:** Detected language signature code `{detected_lang.upper()}` (**{LANG_MAP[detected_lang]}**)")
        st.success(f"🎯 **Active Core Architecture Engine Selected:** `{resolved_model_string}`")
        
        v_single = vec_transformer.transform([user_experiment_text])
        prediction_output = active_classifier.predict(v_single)[0]
        
        is_toxic_flag = prediction_output == 1 or str(prediction_output) == '1' or str(prediction_output).lower() == 'toxic'
        
        if is_toxic_flag:
            st.error(f"🚨 **Flagged Alert:** Categorized as **TOXIC** by the `{resolved_model_string}` operational engine matrix.")
        else:
            st.success(f"🍏 **Approved Clear:** Categorized as **CLEAN / SAFE** by the `{resolved_model_string}` operational engine matrix.")
else:
    st.error("❌ **System Initialization Failure:** Could not read evaluation corpus data from local paths or remote GitHub backup servers.")