"""
=============================================================================
Milestone 1 – Preprocessing Pipeline
Project: Customer Support RAG-Powered Intelligent Chatbot
Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
=============================================================================
Steps:
  1. Load dataset
  2. Text cleaning & normalization
  3. Placeholder standardization
  4. Near-duplicate detection
  5. Tokenization & length analysis
  6. RAG chunk construction (with metadata)
  7. Train / Val / Test split
  8. Save processed outputs
=============================================================================
"""

import re
import json
import unicodedata
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download("punkt", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

OUTPUT_DIR = Path("../data/preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True)


# STEP 1 – Load Dataset
df = pd.read_csv("../data/raw_data/customer_support_dataset.csv")


# STEP 2 – Text Cleaning & Normalization
# Regex patterns compiled once for performance
_MULTI_SPACE   = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HTML_TAG      = re.compile(r"<[^>]+>")
_URL           = re.compile(r"https?://\S+|www\.\S+")
_EMAIL         = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_NON_ASCII     = re.compile(r"[^\x00-\x7F]+")

def clean_text(text: str, *, keep_placeholders: bool = True) -> str:
    """
    Core text cleaner:
      - Unicode normalisation (NFKC)
      - Strip HTML tags
      - Mask URLs and emails with tokens
      - Collapse whitespace
      - Strip leading/trailing whitespace

    Parameters
    ----------
    text             : raw string
    keep_placeholders: if True, {{PLACEHOLDER}} patterns are left intact
                       (they are handled by standardize_placeholders later)
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove HTML tags
    text = _HTML_TAG.sub(" ", text)

    # 3. Mask URLs / emails (preserve semantic meaning with tokens)
    text = _URL.sub("[URL]", text)
    text = _EMAIL.sub("[EMAIL]", text)

    # 4. Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # 5. Strip
    text = text.strip()

    return text


# STEP 3 – Placeholder Standardisation
# Map every known {{placeholder}} variant to a canonical token
PLACEHOLDER_MAP = {
    # Orders
    r"\{\{Order Number\}\}":       "[ORDER_NUMBER]",
    r"\{\{Order ID\}\}":           "[ORDER_NUMBER]",
    r"\{\{order number\}\}":       "[ORDER_NUMBER]",
    r"\{\{order id\}\}":           "[ORDER_NUMBER]",

    # Accounts
    r"\{\{Account Type\}\}":       "[ACCOUNT_TYPE]",
    r"\{\{account type\}\}":       "[ACCOUNT_TYPE]",
    r"\{\{Account Number\}\}":     "[ACCOUNT_NUMBER]",

    # People
    r"\{\{Name\}\}":               "[CUSTOMER_NAME]",
    r"\{\{name\}\}":               "[CUSTOMER_NAME]",
    r"\{\{Agent Name\}\}":         "[AGENT_NAME]",
    r"\{\{Customer Name\}\}":      "[CUSTOMER_NAME]",

    # Dates & times
    r"\{\{Date\}\}":               "[DATE]",
    r"\{\{date\}\}":               "[DATE]",
    r"\{\{Time\}\}":               "[TIME]",
    r"\{\{ETA\}\}":                "[ETA]",

    # Money
    r"\{\{Amount\}\}":             "[AMOUNT]",
    r"\{\{amount\}\}":             "[AMOUNT]",
    r"\{\{Refund Amount\}\}":      "[AMOUNT]",

    # Products
    r"\{\{Product\}\}":            "[PRODUCT]",
    r"\{\{product\}\}":            "[PRODUCT]",
    r"\{\{Product Name\}\}":       "[PRODUCT]",
    r"\{\{Item\}\}":               "[PRODUCT]",

    # Contact / addresses
    r"\{\{Phone Number\}\}":       "[PHONE_NUMBER]",
    r"\{\{Email Address\}\}":      "[EMAIL_ADDRESS]",
    r"\{\{Address\}\}":            "[ADDRESS]",

    # Misc
    r"\{\{Ticket Number\}\}":      "[TICKET_NUMBER]",
    r"\{\{[A-Za-z ]+\}\}":         "[PLACEHOLDER]",   # catch-all
}

def standardize_placeholders(text: str) -> str:
    """Replace all {{...}} template variables with canonical tokens."""
    for pattern, token in PLACEHOLDER_MAP.items():
        text = re.sub(pattern, token, text)
    return text


def extract_placeholder_inventory(df: pd.DataFrame, cols: list) -> dict:
    """Return a frequency count of every {{...}} found in the dataset."""
    raw_ph = re.compile(r"\{\{[^}]+\}\}")
    counts: Counter = Counter()
    for col in cols:
        for val in df[col].dropna():
            counts.update(raw_ph.findall(val))
    return dict(counts.most_common())



# STEP 4 – Near-Duplicate Detection
def _normalise_for_dedup(text: str) -> str:
    """Lower-case and strip punctuation for dedup fingerprinting."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mark_near_duplicates(df: pd.DataFrame, col: str = "instruction") -> pd.DataFrame:
    """
    Add a boolean column `is_near_duplicate` using a normalised hash of `col`.
    First occurrence is kept (is_near_duplicate = False).
    """
    normalised = df[col].apply(_normalise_for_dedup)
    hashes = normalised.apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    is_dup = hashes.duplicated(keep="first")
    df = df.copy()
    df["is_near_duplicate"] = is_dup
    n_dup = is_dup.sum()
    print(f"      Near-duplicate instructions found: {n_dup:,} ({n_dup/len(df)*100:.1f}%)")
    return df



# STEP 5 – Tokenisation & Length Analysis
def token_length(text: str) -> int:
    """Return word-level token count (GPT tokeniser if available)."""
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    if NLTK_AVAILABLE:
        return len(word_tokenize(text))
    # Whitespace fallback
    return len(text.split())


def add_length_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Append token-length columns for each text column."""
    df = df.copy()
    for col in cols:
        df[f"{col}_token_len"] = df[col].apply(token_length)
    return df


def print_length_report(df: pd.DataFrame, cols: list) -> None:
    """Print descriptive statistics for token lengths."""
    print("\n      Token-length statistics:")
    for col in cols:
        lcol = f"{col}_token_len"
        if lcol not in df.columns:
            continue
        s = df[lcol]
        print(f"        {col:20s} | "
              f"min={s.min():4d}  "
              f"p25={int(s.quantile(0.25)):4d}  "
              f"median={int(s.median()):4d}  "
              f"p75={int(s.quantile(0.75)):4d}  "
              f"p95={int(s.quantile(0.95)):4d}  "
              f"max={s.max():4d}")



# STEP 6 – RAG Chunk Construction
def build_rag_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Convert each row into a RAG-ready document chunk with metadata.

    Chunk schema (stored as JSON-lines):
    {
      "chunk_id"    : str,       # stable SHA-based ID
      "text"        : str,       # the text to embed (instruction + response)
      "instruction" : str,       # clean customer query
      "response"    : str,       # clean agent response
      "category"    : str,
      "intent"      : str,
      "flags"       : str,
      "token_len"   : int        # token count of `text`
    }
    """
    chunks = []
    for _, row in df.iterrows():
        instruction = row.get("instruction_clean", row["instruction"])
        response    = row.get("response_clean",    row["response"])

        # Unified embedding text: label-prefixed for better semantic search
        text = f"Customer: {instruction}\nAgent: {response}"

        chunk_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        chunks.append({
            "chunk_id":    chunk_id,
            "text":        text,
            "instruction": instruction,
            "response":    response,
            "category":    row.get("category", ""),
            "intent":      row.get("intent", ""),
            "flags":       row.get("flags", ""),
            "token_len":   token_length(text),
        })
    return chunks


def build_faq_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Deduplicate by intent to produce one canonical FAQ chunk per intent.
    Selects the response with the median token length (most representative).
    """
    faq = []
    for intent, group in df.groupby("intent"):
        # Pick the example closest to the median response length
        med_len = group["response_clean"].apply(len).median()
        idx = (group["response_clean"].apply(len) - med_len).abs().idxmin()
        row = group.loc[idx]

        faq.append({
            "chunk_id":    f"faq_{intent}",
            "text":        f"Q: {row['instruction_clean']}\nA: {row['response_clean']}",
            "instruction": row["instruction_clean"],
            "response":    row["response_clean"],
            "category":    row["category"],
            "intent":      intent,
            "flags":       "",
            "token_len":   token_length(row["response_clean"]),
        })
    return faq



# STEP 7 – Train / Val / Test Split
def stratified_split(
    df: pd.DataFrame,
    stratify_col: str = "intent",
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split that preserves intent distribution.
    Returns (train, val, test).
    """
    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[stratify_col],
        random_state=seed,
    )
    val_adjusted = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_adjusted,
        stratify=train_val[stratify_col],
        random_state=seed,
    )
    print(f"      Split → train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


# STEP 8 – Save Outputs
def save_outputs(
    df_full:   pd.DataFrame,
    train:     pd.DataFrame,
    val:       pd.DataFrame,
    test:      pd.DataFrame,
    chunks:    list[dict],
    faq_chunks:list[dict],
    ph_inventory: dict,
) -> None:
    """Persist all processed artefacts to OUTPUT_DIR."""
    print("\n[8/8] Saving artefacts...")

    # Processed full corpus (Parquet – efficient columnar storage)
    df_full.to_parquet(OUTPUT_DIR / "corpus_full.parquet", index=False)
    print("      ✓ corpus_full.parquet")

    # CSV fallback
    df_full.to_csv(OUTPUT_DIR / "corpus_full.csv", index=False)
    print("      ✓ corpus_full.csv")

    # Splits
    train.to_parquet(OUTPUT_DIR / "corpus_train.parquet", index=False)
    val.to_parquet(OUTPUT_DIR /   "corpus_val.parquet",   index=False)
    test.to_parquet(OUTPUT_DIR /  "corpus_test.parquet",  index=False)
    print("      ✓ corpus_train / val / test .parquet")

    # RAG chunks (JSON-lines – one line per chunk, easy streaming)
    with open(OUTPUT_DIR / "rag_chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"      ✓ rag_chunks.jsonl ({len(chunks):,} chunks)")

    # FAQ chunks
    with open(OUTPUT_DIR / "faq_chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in faq_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"      ✓ faq_chunks.jsonl ({len(faq_chunks):,} FAQ entries)")

    # Placeholder inventory
    with open(OUTPUT_DIR / "placeholder_inventory.json", "w") as f:
        json.dump(ph_inventory, f, indent=2)
    print("      ✓ placeholder_inventory.json")

    # Preprocessing stats summary
    stats = {
        "total_rows": len(df_full),
        "near_duplicates": int(df_full["is_near_duplicate"].sum()),
        "clean_rows": int((~df_full["is_near_duplicate"]).sum()),
        "unique_intents": int(df_full["intent"].nunique()),
        "unique_categories": int(df_full["category"].nunique()),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "rag_chunks": len(chunks),
        "faq_chunks": len(faq_chunks),
        "instruction_token_len": {
            k: float(v) for k, v in
            df_full["instruction_token_len"].describe().items()
        } if "instruction_token_len" in df_full.columns else {},
        "response_token_len": {
            k: float(v) for k, v in
            df_full["response_token_len"].describe().items()
        } if "response_token_len" in df_full.columns else {},
    }
    with open(OUTPUT_DIR / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("      ✓ preprocessing_stats.json")



# MAIN PIPELINE
def run_pipeline():
    print("=" * 65)
    print("  RAG Chatbot – Milestone 1 Preprocessing Pipeline")
    print("=" * 65)

    # 1. Load 
    df = pd.read_csv("../data/raw_data/customer_support_dataset.csv")

    # 2. Placeholder inventory (before cleaning, so raw forms captured) 
    print("\n[2/8] Extracting placeholder inventory...")
    ph_inventory = extract_placeholder_inventory(df, ["instruction", "response"])
    print(f"      Found {len(ph_inventory)} distinct placeholder types")
    for ph, cnt in list(ph_inventory.items())[:10]:
        print(f"        {ph:35s} {cnt:>5,}")

    #  3. Clean text 
    print("\n[3/8] Cleaning text fields...")
    df["instruction_clean"] = df["instruction"].apply(clean_text)
    df["response_clean"]    = df["response"].apply(clean_text)

    # Standardise placeholders
    df["instruction_clean"] = df["instruction_clean"].apply(standardize_placeholders)
    df["response_clean"]    = df["response_clean"].apply(standardize_placeholders)

    # Normalise category / intent to lower-snake_case
    df["category"] = df["category"].str.strip().str.upper()
    df["intent"]   = df["intent"].str.strip().str.lower()

    print("      ✓ Cleaning done")

    #  4. Near-duplicate detection 
    print("\n[4/8] Near-duplicate detection...")
    df = mark_near_duplicates(df, col="instruction_clean")

    #  5. Token-length analysis 
    print("\n[5/8] Computing token lengths...")
    df = add_length_stats(df, ["instruction_clean", "response_clean"])
    print_length_report(df, ["instruction_clean", "response_clean"])

    # Flag unusually long responses (>95th percentile) for manual review
    p95 = df["response_clean_token_len"].quantile(0.95)
    df["is_long_response"] = df["response_clean_token_len"] > p95
    print(f"      Long responses (>p95 = {p95:.0f} tokens): "
          f"{df['is_long_response'].sum():,}")

    #  6. Build RAG chunks 
    print("\n[6/8] Building RAG document chunks...")
    # Use only non-duplicate rows for the primary embedding corpus
    df_clean = df[~df["is_near_duplicate"]].reset_index(drop=True)
    chunks    = build_rag_chunks(df_clean)
    faq_chunks = build_faq_chunks(df_clean)
    print(f"      ✓ {len(chunks):,} RAG chunks  |  {len(faq_chunks):,} FAQ chunks")

    #  7. Stratified split 
    print("\n[7/8] Stratified train/val/test split...")
    train, val, test = stratified_split(df_clean, stratify_col="intent")

    #  8. Save 
    save_outputs(df, train, val, test, chunks, faq_chunks, ph_inventory)

    print("\n" + "=" * 65)
    print("  Pipeline complete. Artefacts in:", OUTPUT_DIR.resolve())
    print("=" * 65)

    return df, train, val, test, chunks, faq_chunks


if __name__ == "__main__":
    run_pipeline()