import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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
    and maps it to appropriate deep learning engine architectures without hardcoded lists.
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
            pass # Move to fallback calculation if text sequence length is too small/ambiguous
            
    # Linguistic Structural Signature Fallback (Runs if package fails)
    text_lower = text.lower().split()
    malay_structural_markers = {"yang", "dan", "di", "ke", "itu", "ini", "untuk", "dengan", "saya", "kamu", "benci"}
    ms_matches = sum(1 for word in text_lower if word in malay_structural_markers)
    return "ms" if ms_matches > 0 else "en"

def preprocess(text, lang):
    return text.strip()

# ── Config & Repository Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
LOCAL_DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_corpus.csv")

# Remote Git URLs targeting your source project repository
REMOTE_DATA_URL_MAIN = "https://raw.githubusercontent.com/lcokun/final-project-nlp/main/balanced_corpus.csv"
REMOTE_DATA_URL_MASTER = "https://raw.githubusercontent.com/lcokun/final-project-nlp/master/balanced_corpus.csv"

HF_REPO_EN = "lcokun/toxic-comment-distilbert"          
HF_REPO_BI = "lcokun/toxic-comment-xlm-roberta-bilingual" 
LOCAL_BILINGUAL = os.path.join(BASE_DIR, "models", "xlm_roberta_bilingual")
MAX_LEN = 128

LANG_MAP = {
    "en": "DistilBERT (English-Only Core Engine)",
    "ms": "XLM-RoBERTa (Bilingual Core Engine)"
}

st.set_page_config(page_title="Transformer Deep Learning Hub", layout="wide")

# University color accents matching your portfolio theme layout guidelines
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

st.title("🔬 Engine 3: Deep Learning Transformer Models Workspace")
st.markdown("---")

# =====================================================================
# DATA & ASSET CACHING LAYER
# =====================================================================
@st.cache_resource
def load_transformer_models():
    """Loads and caches both required deep learning transformers into memory."""
    cache = {}
    with torch.no_grad():
        # 1. Load English-Only Model (DistilBERT)
        en_tokenizer = AutoTokenizer.from_pretrained(HF_REPO_EN)
        en_model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_EN, output_attentions=True)
        en_model.eval()
        cache["DistilBERT (English-Only)"] = {"model": en_model, "tokenizer": en_tokenizer, "lang_key": "en", "repo_id": HF_REPO_EN}

        # 2. Load Bilingual Model (XLM-RoBERTa)
        bi_src = LOCAL_BILINGUAL if os.path.isdir(LOCAL_BILINGUAL) else HF_REPO_BI
        bi_tokenizer = AutoTokenizer.from_pretrained(bi_src)
        bi_model = AutoModelForSequenceClassification.from_pretrained(bi_src, output_attentions=True)
        bi_model.eval()
        cache["XLM-RoBERTa (Bilingual)"] = {"model": bi_model, "tokenizer": bi_tokenizer, "lang_key": "ms", "repo_id": HF_REPO_BI}
    return cache

@st.cache_data
def load_clean_evaluation_data():
    """Loads validation dataset locally, falling back to streaming directly from GitHub repository."""
    if os.path.exists(LOCAL_DATA_PATH):
        return pd.read_csv(LOCAL_DATA_PATH)
        
    for url in [REMOTE_DATA_URL_MAIN, REMOTE_DATA_URL_MASTER]:
        try:
            df = pd.read_csv(url)
            return df
        except Exception:
            continue
    return None

# Load underlying dataset and transformer caches
df_corpus = load_clean_evaluation_data()
with st.spinner("Initializing neural model pipelines and processing checkpoints..."):
    model_cache = load_transformer_models()

if df_corpus is not None and model_cache is not None:
    # Safely identify dataset structural columns
    text_col = "comment_text" if "comment_text" in df_corpus.columns else df_corpus.columns[0]
    label_col = "is_toxic" if "is_toxic" in df_corpus.columns else df_corpus.columns[1]

    # =====================================================================
    # 2. INTERACTIVE SELECTBOX HUB CONTROLS
    # =====================================================================
    st.subheader("⚙️ Classifier Selector Panel")
    chosen_algo = st.selectbox(
        "Choose an Active Deep Learning Transformer Architecture to Evaluate & Review:",
        options=list(model_cache.keys())
    )

    # Resolve active selected assets
    active_assets = model_cache[chosen_algo]
    active_model = active_assets["model"]
    active_tokenizer = active_assets["tokenizer"]
    target_lang_key = active_assets["lang_key"]
    active_repo_id = active_assets["repo_id"]

    # =====================================================================
    # 3. DYNAMIC PERFORMANCE BENCHMARKS GENERATION FOR THE CHOSEN MODEL
    # =====================================================================
    @st.cache_data
    def compile_specific_model_metrics(chosen_model_name):
        # Slice compilation validation sample array size to maintain interactive UI load speeds
        sample_size = min(len(df_corpus), 100)
        eval_df = df_corpus.sample(sample_size, random_state=42)
        
        texts = eval_df[text_col].astype(str).tolist()
        y_true_labels = eval_df[label_col].tolist()
        
        y_pred_labels = []
        predicted_probabilities = []
        
        # Pull specific internal structures
        local_assets = model_cache[chosen_model_name]
        loc_tokenizer = local_assets["tokenizer"]
        loc_model = local_assets["model"]
        loc_lang = local_assets["lang_key"]
        
        for t_item in texts:
            p_text = t_item if (loc_lang == "ms") else preprocess(t_item, loc_lang)
            tokens_in = loc_tokenizer(p_text, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
            
            with torch.no_grad():
                logits_out = loc_model(**tokens_in)
                
            s_probs = torch.softmax(logits_out.logits, dim=-1)[0].cpu().numpy()
            predicted_probabilities.append(float(s_probs[1]))
            y_pred_labels.append(1 if s_probs[1] >= 0.5 else 0)
            
        return np.array(y_true_labels), np.array(y_pred_labels), np.array(predicted_probabilities)

    with st.spinner(f"Compiling evaluation predictions tensor validation slice for {chosen_algo}..."):
        y_true_real, y_pred_real, probs_real = compile_specific_model_metrics(chosen_algo)

    # =====================================================================
    # 4. CHOSEN CLASSIFIER LIVE METRICS DASHBOARD
    # =====================================================================
    st.markdown("---")
    st.write(f"### 📊 Real-Time Metric Performance Dashboard: **{chosen_algo}**")
    
    acc = accuracy_score(y_true_real, y_pred_real)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_real, y_pred_real, average='weighted', zero_division=0)

    # Render summary metrics cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Calculated Global Accuracy", value=f"{acc * 100:.2f}%")
    m2.metric(label="Weighted Class Precision", value=f"{prec * 100:.2f}%")
    m3.metric(label="Weighted Class Recall", value=f"{rec * 100:.2f}%")
    m4.metric(label="Calculated F1-Score Metric", value=f"{f1:.4f}")

    # Detailed report data table 
    per_class_p, per_class_r, per_class_f, _ = precision_recall_fscore_support(y_true_real, y_pred_real, zero_division=0)
    breakdown_df = pd.DataFrame({
        "Precision (Exact Match)": per_class_p,
        "Recall (Catch Bounds)": per_class_r,
        "Calculated F1-Score": per_class_f
    }, index=["Clean / Compliant (Class 0)", "Toxic / Flagged (Class 1)"])
    st.dataframe(breakdown_df.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)

    # =====================================================================
    # 5. THREE DISTINCT DIAGNOSTIC VIZ CHARTS
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Model Diagnostic Graphical Matrix Sheets")
    viz1, viz2, viz3 = st.columns(3)
    
    with viz1:
        st.markdown("##### 1. Confusion Matrix Heatmap")
        cm = confusion_matrix(y_true_real, y_pred_real)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", cbar=False,
                    xticklabels=["Predicted Clean", "Predicted Toxic"],
                    yticklabels=["Actual Clean", "Actual Toxic"], ax=ax_cm)
        plt.title(f"Confusion Matrix Details\n({chosen_algo})", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.caption("Visualization 1: Map checking positive model hits against false indicators.")

    with viz2:
        st.markdown("##### 2. Model Prediction Confidence Spread")
        fig_dist, ax_dist = plt.subplots(figsize=(4.5, 4))
        sns.histplot(probs_real, bins=15, kde=True, color="#6B1F3A", ax=ax_dist)
        ax_dist.set_xlabel("Assigned Toxicity Risk Score Factor")
        ax_dist.set_ylabel("Document Index Counts")
        plt.title("Toxicity Prediction Confidence Spread", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_dist)
        st.caption("Visualization 2: Density layout checking certainty factors mapping model choices.")

    with viz3:
        st.markdown("##### 3. Precision-Recall Curve Boundaries")
        precision_array, recall_array, _ = precision_recall_curve(y_true_real, probs_real)
        fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4))
        ax_pr.plot(recall_array, precision_array, color="#C9922A", lw=2, label="PR Boundary")
        ax_pr.set_xlabel("Recall Factor")
        ax_pr.set_ylabel("Precision Factor")
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlim([0.0, 1.05])
        plt.title("Precision-Recall Structural Curve", fontsize=10, pad=10)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_pr)
        st.caption("Visualization 3: Trade-off calculation checking precision retention as recall bounds scale.")

    # =====================================================================
    # 6. AUTOMATED LANGUAGE ROUTING PLAYGROUND (BOTTOM OF PANEL)
    # =====================================================================
    st.markdown("---")
    st.subheader("🔬 Automated Language Detection & Live Validation Console")
    st.markdown("Type a comment in either **English** or **Malay**. The system will automatically route it through the matching specialized transformer engine.")

    user_experiment_text = st.text_input(
        "Type your validation text statement directly below:",
        value="Saya sangat benci dengan perangai buruk awak.",
        key="transformer_playground_input"
    )

    if st.button("Execute Intelligent Routing Scan") and user_experiment_text.strip() != "":
        # 1. Run automated langdetect checking mechanisms
        detected_lang = detect_language(user_experiment_text)
        
        # 2. Pull correct operational model elements based on live routing choices
        playground_assets = model_cache["XLM-RoBERTa (Bilingual)"] if detected_lang == "ms" else model_cache["DistilBERT (English-Only)"]
        p_tokenizer = playground_assets["tokenizer"]
        p_model = playground_assets["model"]
        p_repo_id = playground_assets["repo_id"]

        # Render Active Routing engine diagnostics alert blocks
        st.info(f"⚡ **Routing Decision Engine:** Detected language code `{detected_lang.upper()}`. Connecting live to architecture target: **{LANG_MAP[detected_lang]}**")
        st.success(f"🎯 **Active Core Architecture Engine Selected:** `{p_repo_id}`")

        # 3. Single sentence matrix inference calculations
        processed_play_text = user_experiment_text if (detected_lang == "ms") else preprocess(user_experiment_text, detected_lang)
        play_inputs = p_tokenizer(processed_play_text, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)

        with torch.no_grad():
            play_outputs = p_model(**play_inputs, output_attentions=True)

        play_probs = torch.softmax(play_outputs.logits, dim=-1)[0].cpu().numpy()
        clean_prob = float(play_probs[0])
        toxic_prob = float(play_probs[1])

        label = "Toxic" if toxic_prob >= 0.5 else "Non-Toxic"
        confidence = toxic_prob * 100 if label == "Toxic" else clean_prob * 100

        # Display Playground Output Fields
        st.markdown("### 🎯 Real-Time Classification Results")
        col_res1, col_res2 = st.columns([2, 3])
        
        with col_res1:
            st.markdown("<br>", unsafe_allow_html=True)
            if label == "Toxic":
                st.error(f"🚨 **Flagged Alert:** Categorized as **TOXIC** by `{p_repo_id}` ({confidence:.2f}% risk weight assigned).")
            else:
                st.success(f"🍏 **Approved Clear:** Categorized as **CLEAN / SAFE** by `{p_repo_id}` ({confidence:.2f}% validation score).")
                
            st.metric(label="Toxicity Confidence Score", value=f"{toxic_prob * 100:.2f}%")
            st.metric(label="Safety/Clean Compliance Score", value=f"{clean_prob * 100:.2f}%")

        with col_res2:
            st.markdown("#### ⚡ Token Attention Alignment Weights")
            last_attn = play_outputs.attentions[-1]
            avg_attn = last_attn[0].mean(dim=0)
            cls_attn = avg_attn[0].cpu().numpy()

            tokens = p_tokenizer.convert_ids_to_tokens(play_inputs["input_ids"][0])
            token_scores = [
                (tok, float(score))
                for tok, score in zip(tokens, cls_attn)
                if tok not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>")
            ]
            token_scores.sort(key=lambda x: x[1], reverse=True)

            if token_scores:
                top_slice = token_scores[:10]
                fig_a, ax_a = plt.subplots(figsize=(7, 3.5))
                ax_a.barh([t[0] for t in top_slice][::-1], [t[1] for t in top_slice][::-1], color="#7F77DD" if detected_lang == "en" else "#E67E22")
                ax_a.set_xlabel("Averaged Attention Interaction Weight (Classification Layer)")
                plt.tight_layout()
                st.pyplot(fig_a)
            else:
                st.caption("Insufficient valid text tokens to plot attention context maps.")
else:
    st.error("❌ **System Initialization Failure:** Could not read evaluation corpus data from local paths or remote GitHub backup servers.")