"""Build the deployable, lyric-derived mood catalog from the trusted local pickle.

The output deliberately uses the term "mood proxy": these values are inferred
from lyrics and are not Spotify Audio Features. Keeping that distinction in the
data makes the offline fallback technically honest and safe to demo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
import re
from pathlib import Path
from urllib.parse import quote

TARGETS = {
    "happy": (0.86, 0.78, 0.82, 0.20, 0.05),
    "sad": (0.18, 0.24, 0.25, 0.76, 0.12),
    "angry": (0.16, 0.90, 0.62, 0.14, 0.04),
    "calm": (0.58, 0.24, 0.30, 0.82, 0.34),
    "fear": (0.22, 0.58, 0.34, 0.44, 0.20),
    "love": (0.80, 0.48, 0.58, 0.54, 0.07),
}
NEUTRAL = (0.50, 0.45, 0.48, 0.50, 0.18)
HEADERS = (
    "id",
    "name",
    "artist",
    "primary_emotion",
    "valence",
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "spotify_url",
    "data_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("df.pkl"))
    parser.add_argument("--model", type=Path, default=Path("emotion_model.pkl"))
    parser.add_argument("--vectorizer", type=Path, default=Path("vectorizer.pkl"))
    parser.add_argument("--labels", type=Path, default=Path("label_encoder.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/tracks.csv"))
    parser.add_argument("--limit", type=int, default=750)
    return parser.parse_args()


def trusted_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - artifacts are versioned locally


def stable_unit(key: str, offset: int) -> float:
    digest = hashlib.sha256(f"{key}:{offset}".encode()).digest()
    return digest[0] / 255.0


def slug_id(artist: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{artist}-{name}".lower()).strip("-")
    suffix = hashlib.sha1(f"{artist}\0{name}".encode()).hexdigest()[:8]  # noqa: S324
    return f"{slug[:42]}-{suffix}"


def proxy_features(emotion: str, confidence: float, key: str) -> list[str]:
    target = TARGETS.get(emotion, NEUTRAL)
    strength = 0.52 + min(max(confidence, 0.0), 1.0) * 0.36
    features: list[str] = []
    for index, (target_value, neutral_value) in enumerate(zip(target, NEUTRAL, strict=False)):
        jitter = (stable_unit(key, index) - 0.5) * 0.12
        value = target_value * strength + neutral_value * (1 - strength) + jitter
        features.append(f"{min(max(value, 0.01), 0.99):.4f}")
    return features


def build_catalog(args: argparse.Namespace) -> int:
    frame = trusted_pickle(args.source)
    classifier = trusted_pickle(args.model)
    vectorizer = trusted_pickle(args.vectorizer)
    label_encoder = trusted_pickle(args.labels)

    frame = frame.dropna(subset=["artist", "song", "text"]).copy()
    frame["artist"] = frame["artist"].astype(str).str.strip()
    frame["song"] = frame["song"].astype(str).str.strip()
    frame = frame[(frame["artist"] != "") & (frame["song"] != "")]
    frame = frame.drop_duplicates(subset=["artist", "song"]).head(args.limit)

    probabilities = classifier.predict_proba(vectorizer.transform(frame["text"].astype(str)))
    predictions = probabilities.argmax(axis=1)
    emotions = label_encoder.inverse_transform(predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        for row, emotion, scores in zip(frame.itertuples(), emotions, probabilities, strict=False):
            artist = row.artist
            name = row.song
            key = f"{artist}\0{name}"
            features = proxy_features(str(emotion), float(max(scores)), key)
            spotify_url = "https://open.spotify.com/search/" + quote(f"{name} {artist}")
            writer.writerow(
                [
                    slug_id(artist, name),
                    name,
                    artist,
                    emotion,
                    *features,
                    spotify_url,
                    "lyrics-mood-proxy-v1",
                ]
            )
    return len(frame)


if __name__ == "__main__":
    arguments = parse_args()
    count = build_catalog(arguments)
    print(f"Wrote {count} tracks to {arguments.output}")
