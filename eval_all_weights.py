"""在 validation 集上评测仓库内全部权重，输出 micro-F1 汇总表。"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
NER = ROOT / "NER"
OUT = ROOT / "_eval_preds"
sys.path.insert(0, str(ROOT / "pj2"))


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


eval_utils = load_mod("eval_utils", ROOT / "pj2" / "_eval_utils.py")
micro_f1 = eval_utils.micro_f1

GOLD = {
    "English": NER / "English" / "validation.txt",
    "Chinese": NER / "Chinese" / "validation.txt",
}


def run_hmm(npz: Path, lang: str, tag: str) -> float:
    hmm = load_mod("hmm", ROOT / "task1_hmm" / "hmm_ner.py")
    val = GOLD[lang]
    out = OUT / tag / f"{lang}_pred.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    sents = hmm.read_tokens_only(val)
    pred = hmm.HMMNER.load(npz).predict_corpus(sents)
    hmm.write_predictions(out, sents, pred)
    return micro_f1(lang, val, out)


def run_crf(pkl: Path, lang: str, tag: str) -> float:
    crf = load_mod("crf", ROOT / "task2_crf" / "crf_ner.py")
    val = GOLD[lang]
    out = OUT / tag / f"{lang}_pred.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    crf.predict_with_model(pkl, val, out)
    return micro_f1(lang, val, out)


def run_t3(pt: Path, lang: str, tag: str, code_path: Path) -> float:
    import torch

    t3 = load_mod(f"t3_{tag}", code_path)
    val = GOLD[lang]
    out = OUT / tag / f"{lang}_pred.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t3.predict_with_checkpoint(pt, val, out, dev)
    return micro_f1(lang, val, out)


def main() -> None:
    jobs: list[tuple[str, str, str, Path, str]] = []
    # (task, lang, label, weight_path, runner_kind)

    for lang in ("English", "Chinese"):
        p = ROOT / "task1_hmm" / "models" / f"{lang}_hmm.npz"
        if p.is_file():
            jobs.append(("Task1-HMM", lang, "task1_hmm/models", p, "hmm"))

    for lang in ("English", "Chinese"):
        p = ROOT / "task2_crf" / "models" / f"{lang}_crf.pkl"
        if p.is_file():
            jobs.append(("Task2-CRF", lang, "task2_crf/models", p, "crf"))

    for lang in ("English", "Chinese"):
        p = ROOT / "task3_transformer_crf" / "checkpoints" / f"{lang}_transformer_crf.pt"
        if p.is_file():
            jobs.append(
                (
                    "Task3",
                    lang,
                    "task3_transformer_crf/checkpoints",
                    p,
                    "t3_task3",
                )
            )

    for lang in ("English", "Chinese"):
        p = ROOT / "pj2" / "part3" / "checkpoints" / f"{lang}_transformer_crf.pt"
        if p.is_file():
            jobs.append(("Task3", lang, "pj2/part3/checkpoints", p, "t3_pj2"))

    print(f"Gold: {GOLD['English'].name} / {GOLD['Chinese'].name}\n")
    print(f"{'任务':<12} {'语言':<8} {'权重位置':<32} {'micro-F1':>8}")
    print("-" * 64)

    results: list[tuple[str, str, str, float]] = []
    for task, lang, loc, wpath, kind in jobs:
        try:
            if kind == "hmm":
                f1 = run_hmm(wpath, lang, f"{kind}_{loc}")
            elif kind == "crf":
                f1 = run_crf(wpath, lang, f"{kind}_{loc}")
            elif kind == "t3_task3":
                code = ROOT / "task3_transformer_crf" / "transformer_crf_ner.py"
                f1 = run_t3(wpath, lang, f"{kind}_{loc}", code)
            else:
                code = ROOT / "pj2" / "part3" / "transformer_crf_ner.py"
                f1 = run_t3(wpath, lang, f"{kind}_{loc}", code)
            print(f"{task:<12} {lang:<8} {loc:<32} {f1:>8.4f}")
            results.append((task, lang, loc, f1))
        except Exception as e:
            print(f"{task:<12} {lang:<8} {loc:<32} {'ERROR':>8}  {e}")

    print("\n说明: 缺失权重未列出 (如 pj2/part1|2 无 models)。")
    print(f"预测缓存目录: {OUT}")


if __name__ == "__main__":
    main()
