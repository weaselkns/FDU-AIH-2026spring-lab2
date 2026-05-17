"""评测工具：从预测文件计算 micro-F1（与 NER/check.py 一致）。"""

from __future__ import annotations

from pathlib import Path

from sklearn import metrics

SORTED_LABELS_ENG = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]

SORTED_LABELS_CHN = [
    "O",
    "B-NAME",
    "M-NAME",
    "E-NAME",
    "S-NAME",
    "B-CONT",
    "M-CONT",
    "E-CONT",
    "S-CONT",
    "B-EDU",
    "M-EDU",
    "E-EDU",
    "S-EDU",
    "B-TITLE",
    "M-TITLE",
    "E-TITLE",
    "S-TITLE",
    "B-ORG",
    "M-ORG",
    "E-ORG",
    "S-ORG",
    "B-RACE",
    "M-RACE",
    "E-RACE",
    "S-RACE",
    "B-PRO",
    "M-PRO",
    "E-PRO",
    "S-PRO",
    "B-LOC",
    "M-LOC",
    "E-LOC",
    "S-LOC",
]


def read_tags(path: Path) -> list[str]:
    tags: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tags.append(line.split()[-1])
    return tags


def micro_f1(language: str, gold_path: Path, pred_path: Path) -> float:
    labels = (SORTED_LABELS_ENG if language == "English" else SORTED_LABELS_CHN)[1:]
    y_true = read_tags(gold_path)
    y_pred = read_tags(pred_path)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"行数不一致: gold={len(y_true)} pred={len(y_pred)} "
            f"({gold_path} vs {pred_path})"
        )
    return float(
        metrics.f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    )
