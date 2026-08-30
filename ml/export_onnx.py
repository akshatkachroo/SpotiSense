"""Export a trained SpotiSense classifier for Transformers.js browser inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REQUIRED_LABELS = {"angry", "calm", "fear", "happy", "love", "sad"}
TOKENIZER_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
MODEL_INPUT_ORDER = ("input_ids", "attention_mask", "token_type_ids")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ml/artifacts/spotisense-emotion"),
        help="Hugging Face model directory produced by train_transformer.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/public/models/spotisense-emotion"),
        help="Transformers.js model directory",
    )
    parser.add_argument(
        "--minimum-f1",
        type=float,
        default=0.50,
        help="Refuse to export an evaluation below this macro-F1",
    )
    parser.add_argument(
        "--allow-low-quality",
        action="store_true",
        help="Bypass the quality gate for development experiments",
    )
    return parser.parse_args()


def validate_artifact(model_path: Path, minimum_f1: float, allow_low_quality: bool) -> float:
    metrics_path = model_path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing evaluation metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    macro_f1 = float(metrics.get("eval_f1", metrics.get("f1", 0)))
    if macro_f1 < minimum_f1 and not allow_low_quality:
        raise ValueError(
            f"macro-F1 {macro_f1:.3f} is below the {minimum_f1:.3f} export gate; "
            "improve the training data or use --allow-low-quality only for local testing"
        )

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    labels = {str(value).lower() for value in config.get("id2label", {}).values()}
    if labels != REQUIRED_LABELS:
        raise ValueError(f"model labels must be {sorted(REQUIRED_LABELS)}, got {sorted(labels)}")
    return macro_f1


def export_model(model_path: Path, output_path: Path, macro_f1: float) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    model.eval()

    onnx_directory = output_path / "onnx"
    onnx_directory.mkdir(parents=True, exist_ok=True)
    full_precision_path = onnx_directory / "model.onnx"
    quantized_path = onnx_directory / "model_quantized.onnx"

    encoded = tokenizer(
        "I feel calm and ready to focus.",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    # Match BertForSequenceClassification.forward's positional argument order.
    # Tokenizer dictionaries can place token_type_ids before attention_mask.
    input_names = [name for name in MODEL_INPUT_ORDER if name in encoded]
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    with torch.inference_mode():
        torch.onnx.export(
            model,
            tuple(encoded[name] for name in input_names),
            full_precision_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    quantize_dynamic(
        model_input=full_precision_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
    )
    full_precision_path.unlink()
    for filename in TOKENIZER_FILES:
        source = model_path / filename
        if source.exists():
            shutil.copy2(source, output_path / filename)

    digest = hashlib.sha256(quantized_path.read_bytes()).hexdigest()
    manifest = {
        "format": "ONNX",
        "quantization": "int8-dynamic",
        "source": str(model_path),
        "macro_f1": macro_f1,
        "labels": sorted(REQUIRED_LABELS),
        "model_bytes": quantized_path.stat().st_size,
        "sha256": digest,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return quantized_path


if __name__ == "__main__":
    arguments = parse_args()
    score = validate_artifact(
        arguments.model, arguments.minimum_f1, arguments.allow_low_quality
    )
    exported = export_model(arguments.model, arguments.output, score)
    print(f"Exported browser model to {exported} ({exported.stat().st_size / 1_000_000:.1f} MB)")
