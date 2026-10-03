"""Fine-tune the BERT mandate extractor on the stored JSONL dataset.

This script trains ONE shared BERT encoder jointly with:
- a BIO token-classification head for NER / slot filling;
- seven request-level sequence-classification heads.

No GridSearch.  Fixed training configuration on a stored, inspectable dataset.

Run once from the project root:
    pip install -r requirements.txt
    python train_nlp.py

The first run downloads the pretrained BERT checkpoint.  The resulting local
checkpoint under models/nlp_mandate/ is self-contained for later inference.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.nlp_model import MandateNLPModel


BASE_MODEL = "bert-base-uncased"
DATA_PATH = Path("data/nlp_training/client_mandate_nlp.jsonl")
MODEL_DIR = Path("models/nlp_mandate")
MAX_LENGTH = 192
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
SEED = 42

ENTITY_TYPES = [
    "UNDERLYING",
    "THEME",
    "TARGET_RETURN",
    "MATURITY",
    "NOTIONAL",
    "CURRENCY",
    "DOWNSIDE_TOLERANCE",
    "CAPITAL_PROTECTION",
    "PRODUCT",
]
TOKEN_LABELS = ["O"] + [tag for ent in ENTITY_TYPES for tag in (f"B-{ent}", f"I-{ent}")]
TOKEN_TO_ID = {label: i for i, label in enumerate(TOKEN_LABELS)}

SEQUENCE_LABELS: Dict[str, List[str]] = {
    "directional_view": ["bearish", "neutral", "bullish"],
    "return_type": ["unspecified", "income", "participation", "binary_payout", "range_income", "capital_protection"],
    "income_preference": ["no", "yes", "not_specified"],
    "autocall_acceptance": ["reject", "accept", "not_specified"],
    "risk_appetite": ["low", "medium", "high"],
    "complexity_tolerance": ["low", "medium", "high"],
    "memory_requested": ["no", "yes", "not_specified"],
}
SEQ_TO_ID = {name: {label: i for i, label in enumerate(labels)} for name, labels in SEQUENCE_LABELS.items()}


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def align_token_labels(offsets, entities) -> List[int]:
    """Convert character spans to BIO labels on BERT subword tokens."""
    result: List[int] = []
    seen_entity = set()

    for token_index, (start, end) in enumerate(offsets):
        if start == end:  # [CLS], [SEP], padding
            result.append(-100)
            continue

        label = "O"
        for entity_index, entity in enumerate(entities):
            if overlaps(start, end, int(entity["start"]), int(entity["end"])):
                prefix = "B" if entity_index not in seen_entity else "I"
                seen_entity.add(entity_index)
                label = f"{prefix}-{entity['label']}"
                break
        result.append(TOKEN_TO_ID[label])
    return result


class MandateDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer) -> None:
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        token_labels = align_token_labels(encoded.pop("offset_mapping"), row["entities"])
        seq = {
            name: SEQ_TO_ID[name][row["sequence_labels"][name]]
            for name in SEQUENCE_LABELS
        }
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_labels": token_labels,
            "sequence_labels": seq,
        }


def make_collate(tokenizer):
    def collate(batch: List[Dict]) -> Dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention, token_labels = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
            token_labels.append(item["token_labels"] + [-100] * pad)
        seq = {
            name: torch.tensor([item["sequence_labels"][name] for item in batch], dtype=torch.long)
            for name in SEQUENCE_LABELS
        }
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "token_labels": torch.tensor(token_labels, dtype=torch.long),
            "sequence_labels": seq,
        }
    return collate


def move(batch: Dict, device: torch.device) -> Dict:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "token_labels": batch["token_labels"].to(device),
        "sequence_labels": {k: v.to(device) for k, v in batch["sequence_labels"].items()},
    }


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    token_correct = token_total = 0
    seq_correct = {name: 0 for name in SEQUENCE_LABELS}
    seq_total = 0

    for raw in loader:
        batch = move(raw, device)
        out = model(**batch)
        token_pred = out["token_logits"].argmax(-1)
        mask = batch["token_labels"] != -100
        token_correct += int((token_pred[mask] == batch["token_labels"][mask]).sum())
        token_total += int(mask.sum())

        for name, logits in out["sequence_logits"].items():
            pred = logits.argmax(-1)
            seq_correct[name] += int((pred == batch["sequence_labels"][name]).sum())
        seq_total += int(batch["input_ids"].size(0))

    metrics = {"token_accuracy": token_correct / max(token_total, 1)}
    metrics.update({f"{name}_accuracy": seq_correct[name] / max(seq_total, 1) for name in SEQUENCE_LABELS})
    return metrics


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    rows = read_jsonl(DATA_PATH)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    encoder = AutoModel.from_pretrained(BASE_MODEL)
    model = MandateNLPModel(
        encoder=encoder,
        n_token_labels=len(TOKEN_LABELS),
        sequence_head_sizes={name: len(labels) for name, labels in SEQUENCE_LABELS.items()},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    collate = make_collate(tokenizer)
    train_loader = DataLoader(MandateDataset(train_rows, tokenizer), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(MandateDataset(val_rows, tokenizer), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(MandateDataset(test_rows, tokenizer), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for raw in train_loader:
            batch = move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(out["loss"].detach().cpu())
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={total_loss / max(len(train_loader), 1):.4f} val={val_metrics}")

    test_metrics = evaluate(model, test_loader, device)
    print("test=", test_metrics)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR / "tokenizer")
    encoder.config.save_pretrained(MODEL_DIR / "encoder_config")
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")

    metadata = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "token_labels": TOKEN_LABELS,
        "sequence_labels": SEQUENCE_LABELS,
        "dataset": str(DATA_PATH),
        "dataset_size": n,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "test_metrics": test_metrics,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved NLP checkpoint to {MODEL_DIR}")


if __name__ == "__main__":
    main()
