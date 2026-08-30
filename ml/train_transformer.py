"""Fine-tune and evaluate a compact transformer for six-way emotion detection."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = ("angry", "calm", "fear", "happy", "love", "sad")


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, label_to_id, max_length: int):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = [label_to_id[label] for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index]) for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("ml/data/goemotions_six.csv"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/spotisense-emotion"))
    parser.add_argument("--model", default="prajjwal1/bert-mini")
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("training CSV must contain text and label columns")
    frame = frame.dropna(subset=["text", "label"]).drop_duplicates().copy()
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"unsupported labels: {', '.join(unknown)}")
    counts = frame["label"].value_counts()
    missing = [label for label in LABELS if counts.get(label, 0) < 4]
    if missing:
        raise ValueError(f"each label needs at least four samples; check: {missing}")
    return frame


def compute_metrics(evaluation) -> dict[str, float]:
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
    }


def save_training_plot(log_history: list[dict], output: Path) -> None:
    train_rows = [row for row in log_history if "loss" in row]
    eval_rows = [row for row in log_history if "eval_loss" in row]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if train_rows:
        axis.plot(
            [row.get("epoch", index) for index, row in enumerate(train_rows)],
            [row["loss"] for row in train_rows],
            marker="o",
            label="train",
        )
    if eval_rows:
        axis.plot(
            [row.get("epoch", index) for index, row in enumerate(eval_rows)],
            [row["eval_loss"] for row in eval_rows],
            marker="o",
            label="validation",
        )
    axis.set(title="SpotiSense transformer fine-tuning", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    frame = validate_data(pd.read_csv(args.data))
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=args.seed,
        stratify=frame["label"],
    )
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    train_dataset = EmotionDataset(
        train_frame["text"],
        train_frame["label"].tolist(),
        tokenizer,
        label_to_id,
        args.max_length,
    )
    validation_dataset = EmotionDataset(
        validation_frame["text"],
        validation_frame["label"].tolist(),
        tokenizer,
        label_to_id,
        args.max_length,
    )

    checkpoints = args.output / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    save_training_plot(trainer.state.log_history, args.output / "training_history.png")

    state = {
        "base_model": args.model,
        "labels": LABELS,
        "label_to_id": label_to_id,
        "metrics": metrics,
        "log_history": trainer.state.log_history,
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
    }
    with (args.output / "training_state.pkl").open("wb") as handle:
        pickle.dump(state, handle)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    arguments = parse_args()
    final_metrics = train(arguments)
    print(json.dumps(final_metrics, indent=2, sort_keys=True))
