# %% Sample verse analysis
# Paste as the next cell after main.py has run.
# Requires in scope: trainer, records_by_book, id2label, _make_ds, taam_display_name

import re
import random
import numpy as np

random.seed(42)

_CONSONANT_RE = re.compile(r'[א-תיִ-פֿ]')
_NIKUD_RE     = re.compile(r'[ְ-ׇּׁׂׅׄ]')


def _place_taam(word_text: str, taam_char: str) -> str:
    """Replace all ta'amim on word_text with taam_char at the standard position.

    Inserts after the last consonant + its nikud (first consonant for yetiv).
    """
    from tikkun_loader import TAAM_RANGE_RE
    clean = TAAM_RANGE_RE.sub('', word_text)
    if taam_char == 'z':          # NO_TAAM — strip marks only
        return clean
    consonants = [m.start() for m in _CONSONANT_RE.finditer(clean)]
    if not consonants:
        return clean + taam_char
    # Yetiv (U+059A) marks the first syllable; everything else marks the last
    target = consonants[0] if taam_char == '֚' else consonants[-1]
    pos = target + 1
    while pos < len(clean) and _NIKUD_RE.match(clean[pos]):
        pos += 1
    return clean[:pos] + taam_char + clean[pos:]


def _build_predicted_verse(rec: dict, word_true: list, word_pred: list) -> str:
    """Reconstruct the verse text with predicted ta'am marks.

    Correct words keep their original text; wrong words get the predicted ta'am
    placed at the standard position.
    """
    from tikkun_loader import verse_word_list
    words = [w for w in verse_word_list(rec["text"]) if w["plain"]]
    parts = []
    for idx, w in enumerate(words):
        if idx >= len(word_pred):
            parts.append(w["text"])
            continue
        if word_true[idx] == word_pred[idx]:
            parts.append(w["text"])
        else:
            pred_char = id2label[word_pred[idx]]
            parts.append(_place_taam(w["text"], pred_char))
    return " ".join(parts)


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
    plain    = rec["tokens"]
    true_n   = [taam_display_name(id2label[l]) for l in word_true]
    pred_n   = [taam_display_name(id2label[p]) for p in word_pred]
    mistakes = [(i, true_n[i], pred_n[i]) for i in range(len(plain)) if true_n[i] != pred_n[i]]

    col_w = [max(len(plain[i]), len(true_n[i]), len(pred_n[i])) + 2 for i in range(len(plain))]

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"{rec['book']} {rec['chapter']}:{rec['verse']}   "
          f"{len(mistakes)} mistake{'s' if len(mistakes) != 1 else ''} / {len(plain)} words")
    print(f"  Orig:  {rec['text']}")
    print(f"  Pred:  {_build_predicted_verse(rec, word_true, word_pred)}")
    print("  Words: " + "  ".join(w.ljust(c) for w, c in zip(plain,  col_w)))
    print("  True:  " + "  ".join(t.ljust(c) for t, c in zip(true_n, col_w)))
    pred_fmt = [
        (f"[{p}]" if p != t else p).ljust(c)
        for p, t, c in zip(pred_n, true_n, col_w)
    ]
    print("  Pred:  " + "  ".join(pred_fmt))
    if mistakes:
        details = ", ".join(f"#{i+1} '{plain[i]}': {t}→{p}" for i, t, p in mistakes)
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
