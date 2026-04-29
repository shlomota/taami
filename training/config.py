from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, str] = {
    "alephbert":   "onlplab/alephbert-base",
    "heBERT":      "avichr/heBERT",
    "mbert":       "bert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
}

# ---------------------------------------------------------------------------
# Book splits
# ---------------------------------------------------------------------------
TRAIN_BOOKS = ["GEN", "EXO", "LEV", "NUM"]
VAL_BOOKS   = ["DEU"]


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str  = "alephbert"
    batch_size:  int   = 16
    num_epochs:  int   = 3
    learning_rate: float = 2e-5
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.06
    eval_steps:  int   = 100
    save_steps:  int   = 500
    max_seq_len: int   = 128      # covers >99% of Torah verses
    early_stop_patience: int = 5
    seed: int = 42
    fp16: bool = True             # set False for CPU-only runs
    use_crf: bool = False         # True → BertCRF with Viterbi decode
