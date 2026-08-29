import pandas as pd

from ml.prepare_goemotions import LABEL_GROUPS, balance, collapse_labels


def source_row(text: str, labels: set[str], unclear: bool = False) -> dict:
    row = {
        label: int(label in labels)
        for label in set().union(*LABEL_GROUPS.values())
    }
    return {"text": text, "example_very_unclear": unclear, **row}


def test_collapse_labels_keeps_only_one_product_group():
    frame = pd.DataFrame(
        [
            source_row("I am furious about this", {"anger", "annoyance"}),
            source_row("I am nervous but grateful", {"nervousness", "gratitude"}),
            source_row("This example is unclear", set(), unclear=True),
        ]
    )

    collapsed = collapse_labels(frame)

    assert collapsed.to_dict(orient="records") == [
        {"text": "I am furious about this", "label": "angry"}
    ]


def test_balance_returns_equal_seeded_classes():
    frame = pd.DataFrame(
        [
            {"text": f"{label} example {index}", "label": label}
            for label in LABEL_GROUPS
            for index in range(120)
        ]
    )

    first = balance(frame, per_class=100, seed=42)
    second = balance(frame, per_class=100, seed=42)

    assert first.equals(second)
    assert first["label"].value_counts().to_dict() == {
        label: 100 for label in LABEL_GROUPS
    }
