# %% Sample verse analysis
# Paste as the next cell after main.py has run.
# Requires in scope: trainer, records_by_book, id2label, _make_ds, taam_display_name

import random
import numpy as np

random.seed(42)


def _predict_records(records: list[dict]) -> list[tuple[list[int], list[int]]]:
    ds = _make_ds(records)
    predictions, label_ids, _ = trainer.predict(ds)
    preds = np.argmax(predictions, axis=-1)
    out = []
    for pred_row, label_row in zip(preds, label_ids):
        word_true, word_pred = [], []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                word_true.append(int(l))
                word_pred.append(int(p))
        out.append((word_true, word_pred))
    return out


def _show_verse(rec: dict, word_true: list[int], word_pred: list[int]) -> None:
    plain   = rec["tokens"]
    true_n  = [taam_display_name(id2label[l]) for l in word_true]
    pred_n  = [taam_display_name(id2label[p]) for p in word_pred]
    mistakes = [(i, true_n[i], pred_n[i]) for i in range(len(plain)) if true_n[i] != pred_n[i]]

    # Column width = widest of: plain word, true label, pred label
    col_w = [max(len(plain[i]), len(true_n[i]), len(pred_n[i])) + 2 for i in range(len(plain))]

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"{rec['book']} {rec['chapter']}:{rec['verse']}   "
          f"{len(mistakes)} mistake{'s' if len(mistakes) != 1 else ''} / {len(plain)} words")
    print(f"  Text:  {rec['text']}")
    print("  Words: " + "  ".join(w.ljust(c) for w, c in zip(plain,  col_w)))
    print("  True:  " + "  ".join(t.ljust(c) for t, c in zip(true_n, col_w)))
    # Bracket wrong predictions so they stand out
    pred_fmt = [
        (f"[{p}]" if p != t else p).ljust(c)
        for p, t, c in zip(pred_n, true_n, col_w)
    ]
    print("  Pred:  " + "  ".join(pred_fmt))

    if mistakes:
        details = ", ".join(
            f"#{i+1} '{plain[i]}': {t} → {p}" for i, t, p in mistakes
        )
        print(f"  Err:   {details}")
    else:
        print("  Err:   (perfect)")


def sample_and_show(book: str, n: int = 5) -> None:
    print(f"\n{'═' * 72}")
    print(f"  SAMPLE VERSES — {book}")
    print(f"{'═' * 72}")
    recs = records_by_book.get(book, [])
    if not recs:
        print(f"  No records for {book}")
        return
    sample = random.sample(recs, min(n, len(recs)))
    for rec, (word_true, word_pred) in zip(sample, _predict_records(sample)):
        _show_verse(rec, word_true, word_pred)


sample_and_show("DEU", n=5)
sample_and_show("LAM", n=5)
