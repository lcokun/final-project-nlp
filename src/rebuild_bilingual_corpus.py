"""
Rebuilds the bilingual training corpus with per-language balancing.

English: data/balanced_corpus.csv (already 50/50, ~30,568 rows, preprocessed)
Malay:   data/Malay Local Reformat.csv (raw comment_text, lang=ms, balanced 50/50)

Malay rows are upsampled to MS_TARGET to reduce EN/MS imbalance.
Output: data/balanced_corpus_fixed.csv with text and lang columns.
"""

import os
import pandas as pd

BASE           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN_PATH        = os.path.join(BASE, "data", "balanced_corpus.csv")
BILINGUAL_PATH = os.path.join(BASE, "data", "Malay Local Reformat.csv")
OUT_PATH       = os.path.join(BASE, "data", "balanced_corpus_fixed.csv")

MS_TARGET = 20000  # upsample Malay to this total (10k per class)
SEED      = 42

# ── English corpus ────────────────────────────────────────────────────────────
print("Loading English corpus...")
df_en = pd.read_csv(EN_PATH).dropna(subset=["preprocessed_text"])
df_en = df_en.rename(columns={"preprocessed_text": "text"})
df_en["lang"] = "en"
print(f"English: {df_en.shape[0]} rows | {df_en['toxic'].value_counts().to_dict()}")

# ── Malay corpus — raw text, no preprocessing ─────────────────────────────────
print("\nLoading Malay corpus...")
df_bilingual = pd.read_csv(BILINGUAL_PATH)
df_ms = df_bilingual[df_bilingual["lang"] == "ms"].copy().reset_index(drop=True)
df_ms = df_ms.dropna(subset=["comment_text"])
df_ms = df_ms.rename(columns={"comment_text": "text"})
print(f"Malay raw: {df_ms.shape[0]} rows | {df_ms['toxic'].value_counts().to_dict()}")

# Balance 50/50 within Malay before upsampling
toxic_ms = df_ms[df_ms["toxic"] == 1]
clean_ms  = df_ms[df_ms["toxic"] == 0]
n_ms = min(len(toxic_ms), len(clean_ms))
df_ms_balanced = pd.concat([
    toxic_ms.sample(n=n_ms, random_state=SEED),
    clean_ms.sample(n=n_ms, random_state=SEED),
])
print(f"Malay balanced: {df_ms_balanced.shape[0]} rows")

# Upsample Malay to MS_TARGET maintaining 50/50
n_per_class = MS_TARGET // 2
df_ms_final = pd.concat([
    df_ms_balanced[df_ms_balanced["toxic"] == 1].sample(n=n_per_class, replace=True, random_state=SEED),
    df_ms_balanced[df_ms_balanced["toxic"] == 0].sample(n=n_per_class, replace=True, random_state=SEED),
])[["text", "toxic", "lang"]]
print(f"Malay upsampled: {df_ms_final.shape[0]} rows")

# ── Merge ─────────────────────────────────────────────────────────────────────
df_combined = pd.concat(
    [df_en[["text", "toxic", "lang"]], df_ms_final],
    ignore_index=True,
)
df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\nCombined: {df_combined.shape[0]} rows")
print(df_combined.groupby(["lang", "toxic"]).size())

df_combined.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")