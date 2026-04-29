"""
Evaluation metrics: per-word accuracy and per-verse exact-match accuracy.
"""

import numpy as np
import pandas as pd

from tikkun_loader import UNICODE_TO_TAAM

NO_TAAM = "z"


def compute_metrics(eval_pred) -> dict:
    """Trainer-compatible metrics: word accuracy and verse exact-match accuracy."""
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true, y_pred = [], []
    verse_correct = verse_total = 0

    for pred_row, label_row in zip(preds, label_ids):
        word_true, word_pred = [], []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                word_true.append(l)
                word_pred.append(p)
        if word_true:
            y_true.extend(word_true)
            y_pred.extend(word_pred)
            verse_total += 1
            if word_true == word_pred:
                verse_correct += 1

    word_acc  = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true) if y_true else 0.0
    verse_acc = verse_correct / verse_total if verse_total else 0.0
    return {
        "word_accuracy":  round(word_acc,  4),
        "verse_accuracy": round(verse_acc, 4),
    }


def evaluate_book(
    book: str,
    records: list[dict],
    predict_fn,       # trainer.predict
    make_dataset_fn,  # callable(records) -> Dataset
) -> dict | None:
    """
    Evaluate one book: returns word accuracy, verse exact-match accuracy,
    and a per-verse DataFrame with columns:
      book, chapter, verse, n_words, n_correct, word_acc, verse_exact,
      word_true (list[int]), word_pred (list[int])
    """
    if not records:
        return None

    ds = make_dataset_fn(records)
    predictions, label_ids, _ = predict_fn(ds)
    preds = np.argmax(predictions, axis=-1)

    verse_rows = []
    for i, rec in enumerate(records):
        word_true, word_pred = [], []
        for p, l in zip(preds[i], label_ids[i]):
            if l != -100:
                word_true.append(int(l))
                word_pred.append(int(p))
        n = len(word_true)
        if n == 0:
            continue
        n_correct = sum(p == t for p, t in zip(word_pred, word_true))
        verse_rows.append({
            "book":        rec["book"],
            "chapter":     rec["chapter"],
            "verse":       rec["verse"],
            "n_words":     n,
            "n_correct":   n_correct,
            "word_acc":    n_correct / n,
            "verse_exact": int(n_correct == n),
            "word_true":   word_true,
            "word_pred":   word_pred,
        })

    df = pd.DataFrame(verse_rows)
    word_acc  = df["n_correct"].sum() / df["n_words"].sum()
    verse_acc = df["verse_exact"].mean()
    print(f"  {book}: {len(df):4d} verses | word_acc={word_acc:.4f} | verse_acc={verse_acc:.4f}")
    return {"book": book, "df": df, "word_acc": word_acc, "verse_acc": verse_acc}


def taam_display_name(ch: str) -> str:
    """Readable taam name for a Unicode char or NO_TAAM sentinel."""
    if ch == NO_TAAM:
        return "none"
    return UNICODE_TO_TAAM.get(ch, repr(ch))
