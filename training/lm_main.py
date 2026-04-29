# %% [markdown]
# # TAAMI — Cantillation via DictaLM 3.0 (LoRA fine-tuning)
#
# Fine-tunes `DictaLM-3.0-1.7B-Instruct` to predict per-word cantillation
# mark names from a nikud-only verse (ta'amim stripped).
#
# **Task**: verse with nikud → space-separated taam names, one per word.
# e.g. input:  "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
#      output: "tipcha munah silluq"
#
# Why taam names rather than the full cantillated Unicode text?
# DictaLM was pre-trained on modern Hebrew — ta'amim (U+0591–U+05AF) are
# almost absent, so the tokenizer shreds them unpredictably. Taam names are
# plain ASCII the model handles perfectly.
#
# **Colab setup:**
#   !git clone https://github.com/shlomota/taami && %cd taami && !git checkout dev
#   !pip install -r training/requirements.txt -r training/requirements_lm.txt -q

# %% Install (Colab — uncomment)
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install",
#                 "-r", "training/requirements.txt",
#                 "-r", "training/requirements_lm.txt", "-q"], check=True)

# %% Imports
import sys, random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

try:
    _TRAINING_DIR = Path(__file__).resolve().parent
    _REPO_ROOT    = _TRAINING_DIR.parent
except NameError:
    _REPO_ROOT    = Path.cwd()
    _TRAINING_DIR = _REPO_ROOT / "training"

for _p in [str(_REPO_ROOT), str(_TRAINING_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import TRAIN_BOOKS, VAL_BOOKS
from tikkun_loader import (
    load_tikkun_data, strip_teamim,
    verse_word_list, UNICODE_TO_TAAM,
)

# %% [markdown]
# ## Configuration

# %% Config
MODEL_ID   = "dicta-il/DictaLM-3.0-1.7B-Instruct"
OUTPUT_DIR = _TRAINING_DIR / "output_lm"
OUTPUT_DIR.mkdir(exist_ok=True)

# LoRA — applied to all linear projections
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
MAX_SEQ_LEN  = 256   # nikud verse + taam names; ~95th pct fits in 128, use 256 for safety
BATCH_SIZE   = 8
GRAD_ACCUM   = 2     # effective batch = 16
NUM_EPOCHS   = 3
LR           = 2e-4
WARMUP_RATIO = 0.05
SEED         = 42

# Load model in 4-bit (QLoRA) to save VRAM — fine on L4, required on T4 for safety
USE_4BIT = True

# Prompt format — ASCII delimiters tokenise predictably across model versions
INSTRUCTION_TEMPLATE = "### Instruction:\n"
RESPONSE_TEMPLATE    = "### Response:\n"

NO_TAAM = "none"

print(f"Model:  {MODEL_ID}")
print(f"Output: {OUTPUT_DIR}")
print(f"Train:  {TRAIN_BOOKS}  Val: {VAL_BOOKS}")
print(f"4-bit:  {USE_4BIT}")

# %% [markdown]
# ## Data

# %% Helper: primary taam name for a word (last taam char found — matches data.py policy)
def word_taam_name(word_text: str) -> str:
    lbl = None
    for ch in word_text:
        if ch in UNICODE_TO_TAAM:
            lbl = ch
    return UNICODE_TO_TAAM[lbl] if lbl else NO_TAAM


# %% Format a verse as an instruction/response pair
def format_example(verse_text: str) -> str | None:
    words = [w for w in verse_word_list(verse_text) if w["plain"]]
    if not words:
        return None
    verse_no_taam = strip_teamim(verse_text)          # nikud kept, ta'amim stripped
    taam_seq      = " ".join(word_taam_name(w["text"]) for w in words)
    return (
        f"{INSTRUCTION_TEMPLATE}"
        f"ציין את הטעם של כל מילה בפסוק:\n"
        f"{verse_no_taam}\n"
        f"{RESPONSE_TEMPLATE}"
        f"{taam_seq}"
    )


# %% Build train / val datasets
corpus = load_tikkun_data()

def _build(books: list[str]) -> list[str]:
    out = []
    for book, chapters in corpus.items():
        if book not in books:
            continue
        for chap, verses in chapters.items():
            for vnum, text in verses.items():
                ex = format_example(text)
                if ex:
                    out.append(ex)
    return out


random.seed(SEED)
train_texts = _build(TRAIN_BOOKS)
val_texts   = _build(VAL_BOOKS)
print(f"Train: {len(train_texts):,} | Val: {len(val_texts):,}")
print("\nSample example:")
print(train_texts[0])

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})

# %% [markdown]
# ## Tokenizer

# %% Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Sanity-check: response template must tokenise to a consistent sequence
resp_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
print(f"Response template → {len(resp_ids)} tokens: {resp_ids}")

# %% [markdown]
# ## Model + LoRA

# %% Load model
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
) if USE_4BIT else None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16 if not USE_4BIT else None,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# %% Apply LoRA
lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# %% [markdown]
# ## Training


# %% Training config
sft_cfg = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=int(len(train_dataset) * NUM_EPOCHS * WARMUP_RATIO / (BATCH_SIZE * GRAD_ACCUM)),
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=20,
    report_to="none",
    seed=SEED,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=MAX_SEQ_LEN,
    response_template=RESPONSE_TEMPLATE,
)

# %% Train
trainer.train()

trainer.model.save_pretrained(str(OUTPUT_DIR / "best_model"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "best_model"))
print(f"\nSaved to {OUTPUT_DIR / 'best_model'}")

# %% [markdown]
# ## Evaluation

# %% Greedy generation for one verse
def predict_taams(verse_text: str, max_new_tokens: int = 80) -> list[str]:
    """Return predicted taam name list for a verse."""
    verse_no_taam = strip_teamim(verse_text)
    prompt = (
        f"{INSTRUCTION_TEMPLATE}"
        f"ציין את הטעם של כל מילה בפסוק:\n"
        f"{verse_no_taam}\n"
        f"{RESPONSE_TEMPLATE}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return generated.strip().split()


# %% Evaluate on a sample of VAL_BOOKS
def evaluate_lm(books: list[str] = VAL_BOOKS, n: int = 200, show: int = 5) -> dict:
    recs = []
    for book, chapters in corpus.items():
        if book not in books:
            continue
        for chap, verses in chapters.items():
            for vnum, text in verses.items():
                words = [w for w in verse_word_list(text) if w["plain"]]
                if words:
                    recs.append({"text": text, "words": words,
                                 "ref": f"{book} {chap}:{vnum}"})

    sample = random.sample(recs, min(n, len(recs)))
    word_correct = word_total = verse_correct = 0

    for i, rec in enumerate(sample):
        true_names = [word_taam_name(w["text"]) for w in rec["words"]]
        pred_names = predict_taams(rec["text"])

        # Align to true length (pad/trim if generation went wrong)
        n_w = len(true_names)
        pred_names = (pred_names + [NO_TAAM] * n_w)[:n_w]

        n_correct = sum(t == p for t, p in zip(true_names, pred_names))
        word_correct += n_correct
        word_total   += n_w
        if n_correct == n_w:
            verse_correct += 1

        if i < show:
            plain = [w["plain"] for w in rec["words"]]
            col_w = [max(len(pl), len(t), len(p)) + 1
                     for pl, t, p in zip(plain, true_names, pred_names)]
            sep = "─" * 60
            print(f"\n{sep}\n{rec['ref']}")
            print("Words: " + "  ".join(w.ljust(c) for w, c in zip(plain,      col_w)))
            print("True:  " + "  ".join(t.ljust(c) for t, c in zip(true_names, col_w)))
            pred_fmt = [(f"[{p}]" if p != t else p).ljust(c)
                        for p, t, c in zip(pred_names, true_names, col_w)]
            print("Pred:  " + "  ".join(pred_fmt))

    word_acc  = word_correct / word_total if word_total else 0.0
    verse_acc = verse_correct / len(sample) if sample else 0.0
    print(f"\n{'='*60}")
    print(f"Eval on {books}  ({len(sample)} verses)")
    print(f"  Word accuracy:  {word_acc:.4f}")
    print(f"  Verse accuracy: {verse_acc:.4f}")
    return {"word_acc": word_acc, "verse_acc": verse_acc}


print("\nEvaluating on DEU (200 verses)…")
evaluate_lm(VAL_BOOKS, n=200, show=5)
