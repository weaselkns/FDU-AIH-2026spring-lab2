"""
一键面试评测（项目根目录执行）

目录约定（把老师发的数据解压到根目录 test/ 即可）::

    test/English/test.txt
    test/Chinese/test.txt

预测写在根目录（与 test 同级）::

    English_pred_task1.txt
    English_pred_task2.txt
    English_pred_task3.txt
    Chinese_pred_task1.txt
    ...

有标签时用 test 当 gold，自动调用 NER/check.py 打分。
无标签（面试真 test）时:  python run_test_eval.py --no-score

其它命令:
  python run_test_eval.py --tasks 3          # 只跑 Task3
  python run_test_eval.py --train-missing    # 缺 HMM/CRF 权重时先训练
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
TEST_ROOT = REPO_ROOT / "test"
NER_ROOT = REPO_ROOT / "NER"

LANGS = ("English", "Chinese")
TASK_SUFFIX = {1: "task1", 2: "task2", 3: "task3"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_test_file(lang: str) -> Path:
    for name in ("test.txt", "validation.txt"):
        p = TEST_ROOT / lang / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"请把测试数据放在:\n"
        f"  {TEST_ROOT / lang / 'test.txt'}\n"
        f"（或 {TEST_ROOT / lang / 'validation.txt'} 用于本地彩排）"
    )


def _pred_path(lang: str, task_id: int) -> Path:
    return REPO_ROOT / f"{lang}_pred_{TASK_SUFFIX[task_id]}.txt"


def _resolve_t3_ckpt(lang: str) -> Path:
    for p in (
        REPO_ROOT / "task3_transformer_crf" / "checkpoints" / f"{lang}_transformer_crf.pt",
        REPO_ROOT / "pj2" / "part3" / "checkpoints" / f"{lang}_transformer_crf.pt",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(f"未找到 Task3 权重: {lang}_transformer_crf.pt（请先训练 Part3）")


def _run_check(language: str, gold: Path, pred: Path) -> None:
    check = _load("ner_check", NER_ROOT / "check.py")
    print(f"\n{'=' * 60}")
    print(f"check.py  {language}  gold={gold.name}  pred={pred.name}")
    print("=" * 60)
    check.check(language, str(gold.resolve()), str(pred.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description="test/ 一键推理 + 根目录写预测 + check.py")
    parser.add_argument("--tasks", default="all", help="all 或 1,2,3")
    parser.add_argument("--no-score", action="store_true", help="测试集无标签，只出预测")
    parser.add_argument("--train-missing", action="store_true", help="缺 HMM/CRF 时先训练")
    parser.add_argument("--device", default="", help="Task3: cuda / cpu")
    args = parser.parse_args()

    tasks = {1, 2, 3} if args.tasks in ("all", "123") else {int(c) for c in args.tasks if c in "123"}

    hmm_dir = REPO_ROOT / "task1_hmm" / "models"
    crf_dir = REPO_ROOT / "task2_crf" / "models"

    if args.train_missing:
        ner = REPO_ROOT / "NER"
        if 1 in tasks:
            hmm = _load("hmm_ner", REPO_ROOT / "task1_hmm" / "hmm_ner.py")
            for lang in LANGS:
                p = hmm_dir / f"{lang}_hmm.npz"
                if not p.is_file():
                    print(f"[训练] HMM {lang} ...")
                    hmm_dir.mkdir(parents=True, exist_ok=True)
                    hmm.train_and_decode_validation(lang, ner, save_model_path=p)
        if 2 in tasks:
            crf = _load("crf_ner", REPO_ROOT / "task2_crf" / "crf_ner.py")
            for lang in LANGS:
                p = crf_dir / f"{lang}_crf.pkl"
                if not p.is_file():
                    print(f"[训练] CRF {lang} ...")
                    crf_dir.mkdir(parents=True, exist_ok=True)
                    crf.train_and_decode_validation(lang, ner, save_model_path=p)

    import torch

    for lang in LANGS:
        test_path = _find_test_file(lang)
        print(f"\n>>> {lang}  输入: {test_path}")

        if 1 in tasks:
            hmm = _load("hmm_ner", REPO_ROOT / "task1_hmm" / "hmm_ner.py")
            model = hmm_dir / f"{lang}_hmm.npz"
            if not model.is_file():
                raise FileNotFoundError(f"缺少 {model}，先运行: python task1_hmm/hmm_ner.py --lang {lang} --save-model-dir task1_hmm/models")
            out = _pred_path(lang, 1)
            sents = hmm.read_tokens_only(test_path)
            hmm.write_predictions(out, sents, hmm.HMMNER.load(model).predict_corpus(sents))
            print(f"    写出 {out.name}")

        if 2 in tasks:
            crf = _load("crf_ner", REPO_ROOT / "task2_crf" / "crf_ner.py")
            model = crf_dir / f"{lang}_crf.pkl"
            if not model.is_file():
                raise FileNotFoundError(f"缺少 {model}，先运行: python task2_crf/crf_ner.py --lang {lang} --save-model-dir task2_crf/models")
            out = _pred_path(lang, 2)
            crf.predict_with_model(model, test_path, out)

        if 3 in tasks:
            t3 = _load("t3", REPO_ROOT / "task3_transformer_crf" / "transformer_crf_ner.py")
            out = _pred_path(lang, 3)
            dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            t3.predict_with_checkpoint(_resolve_t3_ckpt(lang), test_path, out, dev)

        if not args.no_score:
            for tid in sorted(tasks):
                _run_check(lang, test_path, _pred_path(lang, tid))

    print("\n全部完成。")
    if args.no_score:
        print("未打分（--no-score）。预测文件在仓库根目录: *_pred_task*.txt")
    else:
        print("预测文件在仓库根目录；上方为 check.py 的 classification_report。")


if __name__ == "__main__":
    main()
