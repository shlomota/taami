"""
Corpus loading and dataset construction.
Builds HuggingFace Datasets directly from the etnachta tikkun corpus —
no pre-built pkl files required.
"""

import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset

# Allow importing tikkun_loader from either training/ or repo root as CWD.
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _REPO_ROOT = Path.cwd()  # Jupyter / Colab: assume CWD is repo root

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tikkun_loader import load_tikkun_data, verse_word_list, UNICODE_TO_TAAM

NO_TAAM = "z"


def build_label_encoding() -> tuple[dict[str, int], dict[int, str]]:
    """
    Deterministic label encoding: all Unicode taam chars (sorted) + NO_TAAM sentinel.
    Returns (label2id, id2label).  NO_TAAM 'z' is always the last entry.
    """
    chars = sorted(UNICODE_TO_TAAM.keys()) + [NO_TAAM]
    label2id = {ch: i for i, ch in enumerate(chars)}
    id2label  = {i: ch for i, ch in enumerate(chars)}
    return label2id, id2label


def _verse_to_record(
    book: str, chap: str, vnum: str, text: str, label2id: dict
) -> dict | None:
    words = verse_word_list(text)
    tokens, labels = [], []
    for w in words:
        if not w["plain"]:
            continue
        lbl = NO_TAAM
        for ch in w["text"]:
            if ch in UNICODE_TO_TAAM:
                lbl = ch  # last ta'am on the word wins
        tokens.append(w["plain"])
        labels.append(label2id[lbl])
    if not tokens:
        return None
    return {
        "book":    book,
        "chapter": int(chap),
        "verse":   int(vnum),
        "text":    text,
        "tokens":  tokens,
        "labels":  labels,
        "n_words": len(tokens),
    }


def build_records_by_book(label2id: dict) -> tuple[dict[str, list[dict]], list[str]]:
    """
    Returns (records_by_book, sorted_book_list).
    records_by_book maps book code → list of verse record dicts.
    """
    corpus = load_tikkun_data()
    records_by_book: dict[str, list[dict]] = {}
    for book, chapters in corpus.items():
        recs = []
        for chap, verses in chapters.items():
            for vnum, text in verses.items():
                rec = _verse_to_record(book, chap, vnum, text, label2id)
                if rec:
                    recs.append(rec)
        records_by_book[book] = recs
    return records_by_book, sorted(corpus.keys())


def tokenize_and_align(batch: dict, tokenizer, max_seq_len: int) -> dict:
    """
    Tokenize a batch of word-list sentences and align integer labels to subword tokens.
    Non-first subwords and special tokens get label -100 (ignored by loss).
    """
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_len,
    )
    all_labels = []
    for i, word_labels in enumerate(batch["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned, prev_wid = [], None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev_wid:
                aligned.append(word_labels[wid])
            else:
                aligned.append(-100)
            prev_wid = wid
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized


def make_hf_dataset(records: list[dict], tokenizer, max_seq_len: int) -> Dataset:
    df = pd.DataFrame({
        "tokens": [r["tokens"] for r in records],
        "labels": [r["labels"] for r in records],
    })
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.map(
        lambda batch: tokenize_and_align(batch, tokenizer, max_seq_len),
        batched=True,
        remove_columns=["tokens", "labels"],
    )
