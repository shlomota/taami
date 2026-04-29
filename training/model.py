"""
BertCRF: BERT encoder + linear emission layer + CRF decoder.

CRF operates at *word* level (first-subword positions only, identified by
labels != -100), so the CRF never sees non-contiguous padding gaps.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.config = self.encoder.config  # Trainer reads model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        hidden = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        emissions = self.classifier(self.dropout(hidden))  # (B, T, C)

        loss = None
        if labels is not None:
            # CRF loss at word level: select first-subword positions per example.
            # labels != -100 marks exactly those positions; they form a contiguous
            # word sequence when extracted, satisfying torchcrf's mask constraint.
            total = emissions.new_tensor(0.0)
            n = 0
            for i in range(emissions.size(0)):
                word_mask = labels[i] != -100
                if not word_mask.any():
                    continue
                e = emissions[i][word_mask].unsqueeze(0)   # (1, n_words, C)
                l = labels[i][word_mask].unsqueeze(0)       # (1, n_words)
                total -= self.crf(e, l, reduction="mean")
                n += 1
            loss = total / n if n else emissions.sum() * 0

        return TokenClassifierOutput(loss=loss, logits=emissions)

    def viterbi_one_hot(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Viterbi decode at word level; return one-hot logits for compute_metrics."""
        one_hot = torch.zeros_like(emissions)
        for i in range(emissions.size(0)):
            word_mask = labels[i] != -100
            if not word_mask.any():
                continue
            e = emissions[i][word_mask].unsqueeze(0)
            decoded = self.crf.decode(e)[0]
            positions = word_mask.nonzero(as_tuple=True)[0]
            for pos, lbl in zip(positions, decoded):
                one_hot[i, pos, lbl] = 1.0
        return one_hot

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
        (save_dir / "bert_crf_config.json").write_text(
            json.dumps({"num_labels": self.num_labels})
        )
        self.encoder.config.save_pretrained(save_directory)
