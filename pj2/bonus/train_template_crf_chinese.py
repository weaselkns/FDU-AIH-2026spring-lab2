"""
Bonus：按课程给定 template_for_crf.utf8 构造字级观测特征，训练中文线性链 CRF（sklearn-crfsuite）。

与 Part 2 的区别：Part 2 中文用手工设计的特征名（c、p1、n1 等）；此处特征键与取值
严格由模板行（U00…U09、B00…B09）实例化，对应课堂 CRFsuite 分词模板在 NER 上的用法。
解码仍为库内维特比；「手写」体现在模板解析与特征生成，而非自实现 L-BFGS。

用法（仓库根目录）::

    python pj2/bonus/train_template_crf_chinese.py
    python pj2/bonus/train_template_crf_chinese.py --template pj2/bonus/template_for_crf.utf8
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

from sklearn_crfsuite import CRF

BONUS_DIR = Path(__file__).resolve().parent
PJ2_ROOT = BONUS_DIR.parent
sys.path.insert(0, str(PJ2_ROOT / "part2"))
sys.path.insert(0, str(BONUS_DIR))

from crf_ner import (  # noqa: E402
    ner_data_dir,
    read_tagged_corpus,
    read_tokens_only,
    sentence_to_labels,
    sentence_to_tokens,
    write_predictions,
)
from template_crf_features import parse_template, sentence_to_features  # noqa: E402


def build_xy_template(
    sentences: list[list[tuple[str, str]]],
    templates,
) -> tuple[list[list[dict[str, str]]], list[list[str]]]:
    X: list[list[dict[str, str]]] = []
    y: list[list[str]] = []
    for sent in sentences:
        toks = sentence_to_tokens(sent)
        labs = sentence_to_labels(sent)
        if not toks:
            continue
        X.append(sentence_to_features(toks, templates))
        y.append(labs)
    return X, y


def print_check_hint(gold_path: Path, pred_path: Path) -> None:
    gold_rel = os.path.relpath(gold_path, ner_data_dir())
    pred_rel = os.path.relpath(pred_path, ner_data_dir())
    print("\n--- NER/check.py（在 NER 目录下执行）---")
    print(
        "python -c \"from check import check; "
        f"check(language='Chinese', gold_path=r'{gold_rel}', my_path=r'{pred_rel}')\""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonus：模板特征中文 CRF-NER")
    parser.add_argument(
        "--template",
        type=str,
        default=str(BONUS_DIR / "template_for_crf.utf8"),
        help="CRFsuite 风格模板路径",
    )
    parser.add_argument("--c1", type=float, default=0.08)
    parser.add_argument("--c2", type=float, default=0.4)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument(
        "--save-model",
        type=str,
        default="",
        help="若非空，写入该路径（pickle）",
    )
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()

    if args.model_path or args.input_path or args.output_path:
        if not (args.model_path and args.input_path and args.output_path):
            raise SystemExit("仅解码需同时提供 --model-path --input-path --output-path")
        with Path(args.model_path).open("rb") as f:
            bundle = pickle.load(f)
        templates = bundle["templates"]
        crf: CRF = bundle["crf"]
        sents = read_tokens_only(Path(args.input_path))
        X = [sentence_to_features(s, templates) for s in sents]
        y_pred = crf.predict(X)
        write_predictions(Path(args.output_path), sents, y_pred)
        print(f"已写出: {args.output_path}")
        return

    tpl_path = Path(args.template)
    print(f"读取模板: {tpl_path}")
    templates = parse_template(tpl_path)
    print(f"  共 {len(templates)} 条观测模板（U/B）")

    ner_root = ner_data_dir()
    train_path = ner_root / "Chinese" / "train.txt"
    val_path = ner_root / "Chinese" / "validation.txt"
    out_dir = ner_root / "predictions" / "bonus_template_crf"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "Chinese_validation_bonus_template_crf.txt"

    print(f"读训练: {train_path}")
    train_sents = read_tagged_corpus(train_path)
    X_train, y_train = build_xy_template(train_sents, templates)
    print(f"句数={len(X_train)}  模板特征维(每位置键数)≈{len(X_train[0][0]) if X_train else 0}")

    crf = CRF(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
        all_possible_transitions=True,
        verbose=False,
    )
    print("训练 CRF（L-BFGS）…")
    crf.fit(X_train, y_train)

    val_tokens = read_tokens_only(val_path)
    X_val = [sentence_to_features(s, templates) for s in val_tokens]
    print("维特比解码验证集…")
    y_pred = crf.predict(X_val)
    write_predictions(pred_path, val_tokens, y_pred)
    print(f"写出: {pred_path}")
    print_check_hint(val_path, pred_path)

    if args.save_model:
        mp = Path(args.save_model)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("wb") as f:
            pickle.dump({"language": "Chinese", "crf": crf, "templates": templates}, f)
        print(f"模型已保存: {mp}")


if __name__ == "__main__":
    main()
