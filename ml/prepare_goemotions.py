"""Build a balanced six-class training set from Google Research GoEmotions."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://storage.googleapis.com/gresearch/goemotions/"
    "data/full_dataset/goemotions_{part}.csv"
)

LABEL_GROUPS = {
    "angry": {"anger", "annoyance", "disapproval", "disgust"},
    "calm": {"neutral", "realization", "relief"},
    "fear": {"fear", "nervousness"},
    "happy": {"amusement", "excitement", "joy", "optimism", "pride", "surprise"},
    "love": {"admiration", "caring", "desire", "gratitude", "love"},
    "sad": {"disappointment", "embarrassment", "grief", "remorse", "sadness"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("ml/data/goemotions-raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("ml/data/goemotions_six.csv")
    )
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_sources(raw_directory: Path) -> list[Path]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for part in range(1, 4):
        destination = raw_directory / f"goemotions_{part}.csv"
        if not destination.exists():
            request = urllib.request.Request(
                SOURCE_URL.format(part=part),
                headers={"User-Agent": "SpotiSense dataset preparation"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                destination.write_bytes(response.read())
        paths.append(destination)
    return paths


def collapse_labels(frame: pd.DataFrame) -> pd.DataFrame:
    all_source_labels = set().union(*LABEL_GROUPS.values())
    missing = all_source_labels - set(frame.columns)
    if missing:
        raise ValueError(f"GoEmotions source is missing columns: {sorted(missing)}")

    rows: list[dict[str, str]] = []
    for record in frame.to_dict(orient="records"):
        if bool(record.get("example_very_unclear")):
            continue
        active = {label for label in all_source_labels if int(record[label]) == 1}
        groups = [name for name, labels in LABEL_GROUPS.items() if active & labels]
        if len(groups) != 1:
            continue
        text = " ".join(str(record["text"]).split())
        if 5 <= len(text) <= 500:
            rows.append({"text": text, "label": groups[0]})
    return pd.DataFrame(rows).drop_duplicates(subset="text")


def balance(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    if per_class < 100:
        raise ValueError("per-class must be at least 100")
    available = frame["label"].value_counts()
    missing = [label for label in LABEL_GROUPS if available.get(label, 0) < per_class]
    if missing:
        counts = {label: int(available.get(label, 0)) for label in missing}
        raise ValueError(f"not enough examples for requested balance: {counts}")
    balanced = pd.concat(
        [
            frame[frame["label"] == label].sample(n=per_class, random_state=seed)
            for label in LABEL_GROUPS
        ],
        ignore_index=True,
    )
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare(arguments: argparse.Namespace) -> pd.DataFrame:
    paths = download_sources(arguments.raw_dir)
    source = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    prepared = balance(collapse_labels(source), arguments.per_class, arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(arguments.output, index=False)
    return prepared


if __name__ == "__main__":
    args = parse_args()
    dataset = prepare(args)
    print(f"Wrote {len(dataset):,} examples to {args.output}")
    print(dataset["label"].value_counts().sort_index().to_string())
