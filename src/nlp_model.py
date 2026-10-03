"""Shared BERT encoder with two supervised NLP branches.

Architecture
------------
Client request -> BERT encoder ->
    1) token-classification head (BIO NER / slot filling)
    2) request-level sequence-classification heads

The token head extracts spans such as UNDERLYING, MATURITY, NOTIONAL,
TARGET_RETURN and PRODUCT.  The sequence heads classify global mandate intent
such as directional view, income preference and autocall acceptance.

The same encoder is fine-tuned jointly by both branches.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class MandateNLPModel(nn.Module):
    """One shared Transformer encoder with token and sequence heads."""

    def __init__(
        self,
        encoder: nn.Module,
        n_token_labels: int,
        sequence_head_sizes: Dict[str, int],
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.token_classifier = nn.Linear(hidden, n_token_labels)
        self.sequence_classifiers = nn.ModuleDict(
            {name: nn.Linear(hidden, n_classes) for name, n_classes in sequence_head_sizes.items()}
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_labels: Optional[torch.Tensor] = None,
        sequence_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(encoded.last_hidden_state)

        # Token-level head: one label distribution for every BERT subword token.
        token_logits = self.token_classifier(hidden)

        # Sequence-level heads: use BERT's [CLS] representation for the whole request.
        cls_vector = self.dropout(encoded.last_hidden_state[:, 0, :])
        sequence_logits = {
            name: head(cls_vector) for name, head in self.sequence_classifiers.items()
        }

        loss = None
        loss_parts: Dict[str, torch.Tensor] = {}

        if token_labels is not None:
            token_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                token_logits.view(-1, token_logits.size(-1)), token_labels.view(-1)
            )
            loss_parts["token"] = token_loss

        if sequence_labels is not None:
            for name, labels in sequence_labels.items():
                seq_loss = nn.CrossEntropyLoss()(sequence_logits[name], labels)
                loss_parts[name] = seq_loss

        if loss_parts:
            # Equal contribution per task; no hand-tuned weighting layer.
            loss = torch.stack(list(loss_parts.values())).mean()

        return {
            "loss": loss,
            "loss_parts": loss_parts,
            "token_logits": token_logits,
            "sequence_logits": sequence_logits,
        }
