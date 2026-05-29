"""
一键面试评测（项目根目录执行）

测试数据目录（--test-root，默认自动选）::

    pj2_test/english_test.txt   # 扁平命名（本仓库彩排）
    pj2_test/chinese_test.txt

    或 test/English/test.txt、test/Chinese/test.txt

预测写在根目录::

    English_pred_task1.txt  ...  English_pred_bert.txt
    Chinese_pred_task1.txt ...

有标签时自动用输入文件当 gold 调 NER/check.py；无标签: --no-score

常用命令见文件末尾 main 的 epilog；pj2_test 彩排示例::

  python run_test_eval.py --test-root pj2_test
  python run_test_eval.py --test-root pj2_test --tasks 4 --batch-size 32
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
NER_ROOT = REPO_ROOT / "NER"

LANGS = ("English", "Chinese")
TASK_SUFFIX = {1: "task1", 2: "task2", 3: "task3", 4: "bert"}

# pj2_test 扁平文件：english_test.txt / chinese_test.txt
_PJ2_TEST_FLAT = {
    "English": "english_test.txt",
    "Chinese": "chinese_test.txt",
}


def _default_test_root() -> Path:
    pj2 = REPO_ROOT / "pj2_test"
    if pj2.is_dir() and any(pj2.glob("*_test.txt")):
        return pj2
    return REPO_ROOT / "test"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_test_file(test_root: Path, lang: str) -> Path:
    flat = test_root / _PJ2_TEST_FLAT[lang]
    if flat.is_file():
        return flat
    for name in ("test.txt", "validation.txt"):
        p = test_root / lang / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"未找到 {lang} 测试数据，请任选一种布局:\n"
        f"  {test_root / _PJ2_TEST_FLAT[lang]}\n"
        f"  {test_root / lang / 'test.txt'}"
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


def _resolve_bert_ckpt(lang: str) -> Path:
    p = REPO_ROOT / "task3_transformer_crf" / "checkpoints_bert" / f"{lang}_bert_crf.pt"
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"未找到 BERT 权重: {p}\n"
        f"请先训练: python task3_transformer_crf/bert_crf_ner.py --lang {lang} "
        f"--save-dir task3_transformer_crf/checkpoints_bert"
    )


def _parse_tasks(s: str) -> set[int]:
    if s in ("all", "1234"):
        return {1, 2, 3, 4}
    if s in ("123", "all3"):
        return {1, 2, 3}
    out = {int(c) for c in s.replace(",", "") if c in "1234"}
    if not out:
        raise ValueError("--tasks 应为 all / 1234 / 或 1,2,3,4 的组合")
    return out


def _run_check(language: str, gold: Path, pred: Path) -> None:
    check = _load("ner_check", NER_ROOT / "check.py")
    print(f"\n{'=' * 60}")
    print(f"check.py  {language}  gold={gold.name}  pred={pred.name}")
    print("=" * 60)
    check.check(language, str(gold.resolve()), str(pred.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键推理 + 根目录写预测 + check.py",
        epilog=(
            "示例:\n"
            "  python run_test_eval.py --test-root pj2_test\n"
            "  python run_test_eval.py --test-root pj2_test --tasks 4 --device cuda\n"
            "  python run_test_eval.py --test-root test --no-score\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", default="all", help="all(含BERT)=1234，或 1,2,3,4")
    parser.add_argument(
        "--test-root",
        type=str,
        default="",
        help="测试数据目录，默认: 有 pj2_test/*_test.txt 则用 pj2_test，否则 test/",
    )
    parser.add_argument("--no-score", action="store_true", help="测试集无标签，只出预测")
    parser.add_argument("--train-missing", action="store_true", help="缺 HMM/CRF 时先训练")
    parser.add_argument("--device", default="", help="Task3/BERT: cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="Task3/BERT 推理 batch")
    args = parser.parse_args()

    test_root = Path(args.test_root) if args.test_root else _default_test_root()
    if not test_root.is_absolute():
        test_root = REPO_ROOT / test_root
    if not test_root.is_dir():
        raise SystemExit(f"测试目录不存在: {test_root}")

    tasks = _parse_tasks(args.tasks)
    print(f"测试目录: {test_root}")

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
        test_path = _find_test_file(test_root, lang)
        print(f"\n>>> {lang}  输入: {test_path.relative_to(REPO_ROOT)}")

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
            print(f"    Task3 小 Transformer  ckpt={_resolve_t3_ckpt(lang).name}  device={dev}")
            t3.predict_with_checkpoint(
                _resolve_t3_ckpt(lang), test_path, out, dev, batch_size=args.batch_size
            )
            print(f"    写出 {out.name}")

        if 4 in tasks:
            bert = _load("bert_ner", REPO_ROOT / "task3_transformer_crf" / "bert_crf_ner.py")
            out = _pred_path(lang, 4)
            dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            ckpt = _resolve_bert_ckpt(lang)
            print(f"    Task4 BERT+CRF  ckpt={ckpt.name}  device={dev}  (仅 predict，不训练)")
            bert.predict_with_checkpoint(
                ckpt, test_path, out, dev, batch_size=args.batch_size
            )
            print(f"    写出 {out.name}")

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
