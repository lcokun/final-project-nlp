"""
Rebuilds the bilingual training corpus with per-language balancing.

English: data/balanced_corpus.csv (already 50/50, 30,568 rows)
Malay:   Data/Malay Local Reformat.csv filtered to lang=ms, preprocessed, balanced 50/50

Output: Data/balanced_corpus_fixed.csv with lang column preserved.
"""

import os
import pandas as pd
from tqdm import tqdm
from src.preprocess import preprocess

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN_PATH     = os.path.join(BASE, "data",  "balanced_corpus.csv")
BILINGUAL_PATH = os.path.join(BASE, "Data", "Malay Local Reformat.csv")
OUT_PATH    = os.path.join(BASE, "Data",  "balanced_corpus_fixed.csv")

# ── English corpus (already balanced 50/50 from Jigsaw) ─────────────────────
print("Loading English corpus...")
df_en = pd.read_csv(EN_PATH).dropna(subset=["preprocessed_text"])
df_en["lang"] = "en"
print(f"English: {df_en.shape[0]} rows | {df_en['toxic'].value_counts().to_dict()}")

# ── Malay corpus ─────────────────────────────────────────────────────────────
print("\nLoading bilingual source file...")
df_bilingual = pd.read_csv(BILINGUAL_PATH)
df_ms_raw = df_bilingual[df_bilingual["lang"] == "ms"].copy().reset_index(drop=True)
df_ms_raw = df_ms_raw.dropna(subset=["comment_text"])
print(f"Malay raw: {df_ms_raw.shape[0]} rows | {df_ms_raw['toxic'].value_counts().to_dict()}")

print("Preprocessing Malay text...")
tqdm.pandas()
df_ms_raw["preprocessed_text"] = df_ms_raw["comment_text"].progress_apply(
    lambda t: preprocess(t, "ms")
)
df_ms_raw = df_ms_raw.dropna(subset=["preprocessed_text"])
df_ms_raw = df_ms_raw[df_ms_raw["preprocessed_text"].str.strip() != ""]

# Balance Malay 50/50 independently
toxic_ms = df_ms_raw[df_ms_raw["toxic"] == 1]
clean_ms  = df_ms_raw[df_ms_raw["toxic"] == 0]
n_ms = min(len(toxic_ms), len(clean_ms))
toxic_ms = toxic_ms.sample(n=n_ms, random_state=42)
clean_ms  = clean_ms.sample(n=n_ms, random_state=42)
df_ms = pd.concat([toxic_ms, clean_ms])[["preprocessed_text", "toxic", "lang"]]
print(f"Malay balanced: {df_ms.shape[0]} rows | {df_ms['toxic'].value_counts().to_dict()}")

# ── Merge ─────────────────────────────────────────────────────────────────────
df_combined = pd.concat(
    [df_en[["preprocessed_text", "toxic", "lang"]],
     df_ms[["preprocessed_text", "toxic", "lang"]]],
    ignore_index=True
)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined: {df_combined.shape[0]} rows")
print(df_combined.groupby(["lang", "toxic"]).size())

df_combined.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
