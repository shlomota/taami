# %% [markdown]
# # TAAMI — Biblical Cantillation Training
#
# Fine-tunes a Hebrew BERT model for per-word cantillation mark prediction
# using token classification.
#
# **Colab setup:**
# ```bash
# !git clone --recurse-submodules <repo-url>
# %cd taami
# !pip install -r training/requirements.txt -q
# # Then run each cell below, or: !python training/main.py
# ```

# %% Install (Colab — uncomment to run)
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install",
#                 "-r", "training/requirements.txt", "-q"], check=True)

# %% Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # headless; comment out for inline Colab display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)

# Add repo root to path so tikkun_loader and training modules are importable.
try:
    _TRAINING_DIR = Path(__file__).resolve().parent
    _REPO_ROOT    = _TRAINING_DIR.parent
except NameError:
    _REPO_ROOT    = Path.cwd()           # Colab: run from repo root
    _TRAINING_DIR = _REPO_ROOT / "training"

for _p in [str(_REPO_ROOT), str(_TRAINING_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config  import TrainConfig, MODEL_REGISTRY, TRAIN_BOOKS, VAL_BOOKS
from data    import build_label_encoding, build_records_by_book, make_hf_dataset
from metrics import compute_metrics, evaluate_book, taam_display_name

# %% [markdown]
# ## Configuration — change `model_name` to try different models

# %% Config
cfg        = TrainConfig(model_name="alephbert")   # see config.MODEL_REGISTRY
model_id   = MODEL_REGISTRY[cfg.model_name]
OUTPUT_DIR = _TRAINING_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Model : {model_id}")
print(f"Train : {TRAIN_BOOKS}")
print(f"Val   : {VAL_BOOKS}")
print(f"Output: {OUTPUT_DIR}")

# %% [markdown]
# ## Labels

# %% Labels
label2id, id2label = build_label_encoding()
num_labels = len(label2id)
print(f"Num labels: {num_labels}")
for i, ch in id2label.items():
    print(f"  {i:2d}  {ch!r}  {taam_display_name(ch)}")

# Human-readable names for the model card (cosmetic only)
_display = {i: taam_display_name(ch) for i, ch in id2label.items()}

# %% [markdown]
# ## Load corpus and build per-book verse records

# %% Data
records_by_book, all_books = build_records_by_book(label2id)

for b in all_books:
    split = "train" if b in TRAIN_BOOKS else ("val" if b in VAL_BOOKS else "other")
    print(f"  {b:5s} [{split:5s}]: {len(records_by_book[b]):4d} verses")

train_records = [r for b in TRAIN_BOOKS for r in records_by_book.get(b, [])]
val_records   = [r for b in VAL_BOOKS   for r in records_by_book.get(b, [])]
print(f"\nTrain total: {len(train_records):,} verses")
print(f"Val   total: {len(val_records):,} verses")

# %% [markdown]
# ## Tokenization

# %% Tokenizer and HF datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)

_make_ds = lambda recs: make_hf_dataset(recs, tokenizer, cfg.max_seq_len)

print("Tokenising train set...")
train_dataset = _make_ds(train_records)
print("Tokenising val set...")
val_dataset   = _make_ds(val_records)
print(f"Train dataset: {len(train_dataset)} examples")
print(f"Val   dataset: {len(val_dataset)} examples")

# %% [markdown]
# ## Model

# %% Model
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=_display,
    label2id={v: k for k, v in _display.items()},
    ignore_mismatched_sizes=True,
)

# %% [markdown]
# ## Training

# %% Training args
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="steps",
    eval_steps=cfg.eval_steps,
    save_strategy="steps",
    save_steps=cfg.save_steps,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="word_accuracy",
    greater_is_better=True,
    learning_rate=cfg.learning_rate,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.num_epochs,
    weight_decay=cfg.weight_decay,
    warmup_ratio=cfg.warmup_ratio,
    fp16=cfg.fp16 and torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=cfg.seed,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stop_patience)],
)

# %% Train
trainer.train()

trainer.save_model(str(OUTPUT_DIR / "best_model"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "best_model"))   # noqa: explicit save
print(f"\nModel saved to {OUTPUT_DIR / 'best_model'}")

# %% [markdown]
# ## Per-book evaluation

# %% Evaluate all books
print("\nEvaluating all books:")
book_results: dict[str, dict] = {}
for book in all_books:
    recs = records_by_book.get(book, [])
    result = evaluate_book(book, recs, trainer.predict, _make_ds)
    if result:
        book_results[book] = result

# %% [markdown]
# ## Plots

# %% Bar chart: word and verse accuracy per book
books_sorted = [b for b in all_books if b in book_results]
word_accs    = [book_results[b]["word_acc"]  for b in books_sorted]
verse_accs   = [book_results[b]["verse_acc"] for b in books_sorted]

x = np.arange(len(books_sorted))
w = 0.35
fig, ax = plt.subplots(figsize=(max(10, len(books_sorted)), 5))
ax.bar(x - w/2, word_accs,  w, label="Word accuracy",  color="steelblue")
ax.bar(x + w/2, verse_accs, w, label="Verse accuracy", color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(books_sorted)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title(f"{cfg.model_name} — per-book accuracy (train={TRAIN_BOOKS}, val={VAL_BOOKS})")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_per_book.png", dpi=150)
plt.close()
print("Saved accuracy_per_book.png")

# %% Confusion matrix on val set
all_true, all_pred = [], []
for b in VAL_BOOKS:
    if b not in book_results:
        continue
    for row in book_results[b]["df"].itertuples():
        all_true.extend(row.word_true)
        all_pred.extend(row.word_pred)

if all_true:
    used_ids    = sorted(set(all_true) | set(all_pred))
    cm          = confusion_matrix(all_true, all_pred, labels=used_ids)
    cm_norm     = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    tick_labels = [taam_display_name(id2label[i]) for i in used_ids]

    n = len(used_ids)
    sz = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(sz, sz))
    sns.heatmap(
        cm_norm, annot=(n <= 30), fmt=".2f", cmap="Blues",
        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (normalised) — {VAL_BOOKS}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_val.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix_val.png")

# %% Save evaluation summary CSV
rows = [
    {
        "book":      b,
        "n_verses":  len(book_results[b]["df"]),
        "word_acc":  round(book_results[b]["word_acc"],  4),
        "verse_acc": round(book_results[b]["verse_acc"], 4),
        "split":     "train" if b in TRAIN_BOOKS else ("val" if b in VAL_BOOKS else "other"),
    }
    for b in books_sorted
]
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUTPUT_DIR / "eval_summary.csv", index=False)
print("\nEvaluation summary:")
print(summary_df.to_string(index=False))
print(f"\nAll outputs saved to {OUTPUT_DIR}")
