Bilingual Malay-English Social Media Dataset for Binary Hate Speech Detection

## Description
This dataset contains 26,985 social media posts in English and Malay, labeled for binary hate speech classification (0 = non-hate, 1 = hate). It is curated from five public datasets: HateXplain, HateM, Toxicity-Small, Snapshot-Twitter-2022, and Supervised-Twitter.

## File
- bilingual_hatespeech_ms_en.csv （full bilingual dataset)
- label_lang_distribution.csv (overall language/label distribution)
- cross-tabulated_distribution.csv (language x source x label breakdown)
- binary_performance.csv (benchmark model results)
- malayslangdict.py (custom Malay slang keyword mappings used for preprocessing)

## Columns
- text: Cleaned social media post
- label: Binary label (0 = non-hate, 1 = hate)
- lang: Language of the text (`en` for English, `ms` for Malay)
- source: Original dataset source (e.g., HateM, HateXplain, Snapshot)

## Preprocessing
Malay texts are cleaned using Malaya NLP (normalisation, slang correction, translation), while English texts are cleaned using Ekphrasis (emoji handling, Twitter symbol normalisation, slang correction). Low-quality texts are removed.
