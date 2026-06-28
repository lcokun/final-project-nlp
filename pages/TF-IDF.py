import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve

# Dynamic folder directory mapping configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_corpus.csv")

st.set_page_config(page_title="TF-IDF Model Diagnostic Workspace", layout="wide")

# University color styling attributes (Maroon & Gold Theme profiles)
st.markdown("""
    <style>
    h1 { color: #6B1F3A; font-family: 'Playfair Display', serif; font-weight: 900; }
    h3, h4 { color: #4a1228; font-family: 'Playfair Display', serif; }
    .stButton>button { background-color: #6B1F3A; color: white; border-radius: 4px; }
    .stButton>button:hover { background-color: #C9922A; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Engine 1: TF-IDF Live Evaluation Hub")
st.markdown("---")

# =====================================================================
# 1. READ CLEAN DATASET & ASSETS SAFELY FROM LOCAL DISK
# =====================================================================
@st.cache_data
def load_clean_corpus_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ **Data File Path Resolution Error:** Could not find `balanced_corpus.csv` inside `{os.path.dirname(DATA_PATH)}`")
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_tfidf_model_assets():
    try:
        # Check if files actually exist first to give you a clear warning
        files = {
            "Naive Bayes (TF-IDF)": "naive_bayes_tfidf.joblib",
            "Logistic Regression (TF-IDF)": "logistic_regression_tfidf.joblib",
            "Support Vector Machine (TF-IDF)": "svm_tfidf.joblib",
            "vec": "vectorizer_tfidf.joblib"
        }
        
        loaded_assets = {}
        for key, filename in files.items():
            full_path = os.path.join(MODELS_PATH, filename)
            if not os.path.exists(full_path):
                st.error(f"❌ Missing expected model file: `{filename}` inside `{MODELS_PATH}`")
                return None
            loaded_assets[key] = joblib.load(full_path)
            
        return loaded_assets
    except Exception as e:
        st.error(f"❌ **Asset Ingestion Error:** Failed to load your scikit-learn TF-IDF `.joblib` pipelines. Details: {e}")
        return None

df_corpus = load_clean_corpus_data()
assets = load_tfidf_model_assets()

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
    
    # DYNAMIC FILTERING: Extract options straight from the asset dictionary keys (excluding the vectorizer)
    available_models = [key for key in assets.keys() if key != "vec"]
    
    chosen_algo = st.selectbox(
        "Choose an Active TF-IDF Classifier to Evaluate & Review:", 
        options=available_models
    )

    # 3. LIVE MATRIX INTERACTION COMPILATION
    vec_transformer = assets["vec"]
    active_classifier = assets[chosen_algo] # This will never throw a KeyError now!

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
    # 6. THREE DISTINCT VIZ CHARTS (Fulfilling Component 4 Rubric Criteria)
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
        # Extract operational probability parameters based on function availability
        if hasattr(active_classifier, "predict_proba"):
            probabilities_array = active_classifier.predict_proba(X_test_vec)[:, 1]
        else:
            # Fallback distance ceiling scaling array logic for LinearSVC profiles
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
        # Computes continuous curve points live
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
    user_experiment_text = st.text_input("Type an experimental sentence structure here to scan with this specific model setup:")
    
    if st.button("Execute Safety Verification Scan") and user_experiment_text.strip() != "":
        v_single = vec_transformer.transform([user_experiment_text])
        prediction_output = active_classifier.predict(v_single)[0]
        
        is_toxic_flag = prediction_output == 1 or str(prediction_output) == '1' or str(prediction_output).lower() == 'toxic'
        
        if is_toxic_flag:
            st.error(f"🚨 **Flagged Alert:** Categorized as **TOXIC** by the {chosen_algo} operational engine matrix.")
        else:
            st.success(f"🍏 **Approved Clear:** Categorized as **CLEAN / SAFE** by the {chosen_algo} operational engine matrix.")