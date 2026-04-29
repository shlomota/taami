"""
Sanity test — run from repo root:
    python tst.py

Checks:
  1. Corpus loads correctly
  2. Label encoding is sane
  3. Train/val split sizes are correct
  4. Tokenisation works
  5. Training loop starts (max_steps=2, CPU)

Expected runtime: ~2–5 minutes on CPU (model download on first run included).
"""

import sys
from pathlib import Path

_REPO_ROOT    = Path(__file__).resolve().parent
_TRAINING_DIR = _REPO_ROOT / "training"

for _p in [str(_REPO_ROOT), str(_TRAINING_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 1. Corpus loading ─────────────────────────────────────────────────────────
print("1. Loading corpus...")
from tikkun_loader import load_tikkun_data, verse_word_list, UNICODE_TO_TAAM

corpus = load_tikkun_data()
assert "GEN" in corpus, "GEN missing from corpus"
assert "DEU" in corpus, "DEU missing from corpus"
print(f"   Books: {sorted(corpus.keys())}")

gen_verses = sum(len(v) for v in corpus["GEN"].values())
print(f"   GEN verse count: {gen_verses}")
assert gen_verses > 1000, f"Expected >1000 GEN verses, got {gen_verses}"

sample_text = list(corpus["GEN"]["1"].values())[0]
words = verse_word_list(sample_text)
assert words, "verse_word_list returned empty for GEN 1:1"
assert all("plain" in w and "teamim" in w for w in words)
print(f"   GEN 1:1: {len(words)} words — taams: {[w['teamim'] for w in words]}")

# ── 2. Label encoding ─────────────────────────────────────────────────────────
print("2. Building label encoding...")
from data import build_label_encoding

label2id, id2label = build_label_encoding()
print(f"   Num labels: {len(label2id)}")
assert len(label2id) >= 27, f"Expected >=27 labels, got {len(label2id)}"
assert "z" in label2id, "NO_TAAM sentinel 'z' missing"
assert label2id["z"] == len(label2id) - 1, "'z' must be the last label"
assert all(ch in label2id for ch in UNICODE_TO_TAAM), "Some taam chars missing from label2id"

# ── 3. Dataset split sizes ────────────────────────────────────────────────────
print("3. Building records by book...")
from data import build_records_by_book
from config import TRAIN_BOOKS, VAL_BOOKS, MODEL_REGISTRY, TrainConfig

records_by_book, all_books = build_records_by_book(label2id)
train_records = [r for b in TRAIN_BOOKS for r in records_by_book.get(b, [])]
val_records   = [r for b in VAL_BOOKS   for r in records_by_book.get(b, [])]

print(f"   Train {TRAIN_BOOKS}: {len(train_records):,} verses")
print(f"   Val   {VAL_BOOKS}: {len(val_records):,} verses")
assert len(train_records) > 3000, f"Expected >3000 train verses, got {len(train_records)}"
assert len(val_records)   > 500,  f"Expected >500 val verses, got {len(val_records)}"

rec = train_records[0]
assert all(k in rec for k in ("book", "chapter", "verse", "tokens", "labels", "n_words"))
assert rec["n_words"] == len(rec["tokens"]) == len(rec["labels"])
assert all(l in id2label for l in rec["labels"]), "Label index out of range in a record"
print(f"   Sample record: book={rec['book']}, verse={rec['chapter']}:{rec['verse']}, "
      f"n_words={rec['n_words']}, labels={rec['labels'][:4]}...")

# ── 4. Tokenisation ───────────────────────────────────────────────────────────
print("4. Testing tokenisation (first 20 verses)...")
from data import make_hf_dataset
from transformers import AutoTokenizer

cfg      = TrainConfig()
model_id = MODEL_REGISTRY[cfg.model_name]
print(f"   Downloading tokenizer: {model_id} (may take a moment on first run)")
tokenizer = AutoTokenizer.from_pretrained(model_id)

sample_ds = make_hf_dataset(train_records[:20], tokenizer, cfg.max_seq_len)
assert len(sample_ds) == 20
assert "input_ids" in sample_ds.column_names
assert "labels"    in sample_ds.column_names
print(f"   Columns: {sample_ds.column_names}")
print(f"   First example input_ids length: {len(sample_ds[0]['input_ids'])}")

# ── 5. Training loop (2 steps, CPU) ──────────────────────────────────────────
print("5. Starting training loop (max_steps=2, CPU)...")
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from metrics import compute_metrics, taam_display_name

_display = {i: taam_display_name(ch) for i, ch in id2label.items()}

print(f"   Downloading model: {model_id}")
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=len(label2id),
    id2label=_display,
    label2id={v: k for k, v in _display.items()},
    ignore_mismatched_sizes=True,
)

val_ds = make_hf_dataset(val_records[:10], tokenizer, cfg.max_seq_len)

args = TrainingArguments(
    output_dir="/tmp/taami_tst",
    max_steps=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=1,
    report_to="none",
    fp16=False,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=sample_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
print("   Training loop ran successfully")

# Quick eval to make sure predict works too
metrics = trainer.evaluate()
print(f"   Eval metrics: {metrics}")

print("\nAll tests passed!")
