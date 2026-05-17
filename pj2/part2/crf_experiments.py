"""
任务二 CRF 批量实验：扫描 L1/L2 正则，记录验证集 micro-F1 并出图。

在仓库根目录执行：``python pj2/part2/crf_experiments.py``（依赖同级 ``NER/`` 数据）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PJ2_ROOT = Path(__file__).resolve().parent.parent
PART2 = Path(__file__).resolve().parent
sys.path.insert(0, str(PJ2_ROOT))
sys.path.insert(0, str(PART2))

from _eval_utils import micro_f1
from crf_ner import ner_data_dir, train_and_decode_validation

FIG_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["English", "Chinese"]
C1_LIST = [0.0, 0.05, 0.08, 0.12, 0.2]
C2_LIST = [0.1, 0.2, 0.4, 0.8, 1.2]
BASE_C2 = 0.4
BASE_C1 = 0.08
MAX_ITERATIONS = 80


def main() -> None:
    ner_root = ner_data_dir()
    exp_pred_dir = FIG_DIR / "preds"
    exp_pred_dir.mkdir(parents=True, exist_ok=True)

    c1_f1: dict[str, np.ndarray] = {}
    c2_f1: dict[str, np.ndarray] = {}

    for lang in LANGUAGES:
        print(f"\n=== CRF {lang}: sweep c1 (c2={BASE_C2}) ===")
        f1_c1: list[float] = []
        for c1 in C1_LIST:
            out_dir = exp_pred_dir / f"{lang}_c1_{c1:g}"
            pred_path, gold_path, _ = train_and_decode_validation(
                language=lang,
                ner_root=ner_root,
                out_dir=out_dir,
                c1=c1,
                c2=BASE_C2,
                max_iterations=MAX_ITERATIONS,
            )
            f1 = micro_f1(lang, gold_path, pred_path)
            f1_c1.append(f1)
            print(f"  c1={c1:g}  micro-F1={f1:.4f}")
        c1_f1[lang] = np.array(f1_c1, dtype=np.float64)

        print(f"\n=== CRF {lang}: sweep c2 (c1={BASE_C1}) ===")
        f1_c2: list[float] = []
        for c2 in C2_LIST:
            out_dir = exp_pred_dir / f"{lang}_c2_{c2:g}"
            pred_path, gold_path, _ = train_and_decode_validation(
                language=lang,
                ner_root=ner_root,
                out_dir=out_dir,
                c1=BASE_C1,
                c2=c2,
                max_iterations=MAX_ITERATIONS,
            )
            f1 = micro_f1(lang, gold_path, pred_path)
            f1_c2.append(f1)
            print(f"  c2={c2:g}  micro-F1={f1:.4f}")
        c2_f1[lang] = np.array(f1_c2, dtype=np.float64)

    np.savez(
        RESULTS_DIR / "crf_results.npz",
        c1_list=np.array(C1_LIST, dtype=np.float64),
        c2_list=np.array(C2_LIST, dtype=np.float64),
        English_c1_f1=c1_f1["English"],
        Chinese_c1_f1=c1_f1["Chinese"],
        English_c2_f1=c2_f1["English"],
        Chinese_c2_f1=c2_f1["Chinese"],
    )

    # c1 曲线
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes, LANGUAGES):
        y = c1_f1[lang]
        ax.plot(range(len(C1_LIST)), y, "s-", linewidth=2, markersize=7, color="C0")
        ax.set_xticks(range(len(C1_LIST)))
        ax.set_xticklabels([f"{c:g}" for c in C1_LIST])
        ax.set_xlabel(f"c1 (c2={BASE_C2})")
        ax.set_ylabel("micro-F1")
        ax.set_title(lang)
        ax.grid(True, alpha=0.3)
    fig.suptitle("CRF: L1 (c1) vs validation micro-F1")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crf_c1_curves.png", dpi=150)
    plt.close(fig)

    # c2 曲线
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes2, LANGUAGES):
        y = c2_f1[lang]
        ax.plot(range(len(C2_LIST)), y, "^-", linewidth=2, markersize=7, color="C1")
        ax.set_xticks(range(len(C2_LIST)))
        ax.set_xticklabels([f"{c:g}" for c in C2_LIST])
        ax.set_xlabel(f"c2 (c1={BASE_C1})")
        ax.set_ylabel("micro-F1")
        ax.set_title(lang)
        ax.grid(True, alpha=0.3)
    fig2.suptitle("CRF: L2 (c2) vs validation micro-F1")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "crf_c2_curves.png", dpi=150)
    plt.close(fig2)

    # 柱状：各语言 c1/c2 扫描最优
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    best_c1 = [c1_f1[l].max() for l in LANGUAGES]
    best_c2 = [c2_f1[l].max() for l in LANGUAGES]
    x = np.arange(len(LANGUAGES))
    w = 0.35
    ax3.bar(x - w / 2, best_c1, w, label="best over c1 sweep")
    ax3.bar(x + w / 2, best_c2, w, label="best over c2 sweep")
    ax3.set_xticks(x)
    ax3.set_xticklabels(LANGUAGES)
    ax3.set_ylabel("micro-F1")
    ax3.set_ylim(0, 1.0)
    ax3.legend()
    ax3.set_title("CRF: best micro-F1 per language (c1 vs c2 sweeps)")
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "crf_best_f1_bar.png", dpi=150)
    plt.close(fig3)

    print(f"\n已保存: {RESULTS_DIR}/crf_results.npz 与 {FIG_DIR}/*.png")


if __name__ == "__main__":
    main()
