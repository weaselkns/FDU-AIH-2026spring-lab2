"""
任务一 HMM 批量实验：扫描发射平滑强度 emit_smoothing，记录验证集 micro-F1 并出图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pj2"))

from pj2._eval_utils import micro_f1
from task1_hmm.hmm_ner import ner_data_dir, train_and_decode_validation

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMIT_SMOOTHINGS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
LANGUAGES = ["English", "Chinese"]


def main() -> None:
    ner_root = ner_data_dir()
    exp_pred_dir = FIG_DIR / "preds"
    exp_pred_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, np.ndarray] = {}
    for lang in LANGUAGES:
        f1_list: list[float] = []
        print(f"\n=== HMM {lang} ===")
        for emit in EMIT_SMOOTHINGS:
            out_dir = exp_pred_dir / f"{lang}_emit{emit:g}"
            pred_path, gold_path, _ = train_and_decode_validation(
                language=lang,
                ner_root=ner_root,
                out_dir=out_dir,
                emit_smoothing=emit,
                trans_smoothing=1e-3,
                init_smoothing=1e-3,
            )
            f1 = micro_f1(lang, gold_path, pred_path)
            f1_list.append(f1)
            print(f"  emit_smoothing={emit:g}  micro-F1={f1:.4f}")
        results[f"{lang}_f1"] = np.array(f1_list, dtype=np.float64)

    np.savez(
        FIG_DIR / "hmm_results.npz",
        emit_smoothings=np.array(EMIT_SMOOTHINGS, dtype=np.float64),
        **results,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes, LANGUAGES):
        f1 = results[f"{lang}_f1"]
        ax.plot(range(len(EMIT_SMOOTHINGS)), f1, "o-", linewidth=2, markersize=7)
        ax.set_xticks(range(len(EMIT_SMOOTHINGS)))
        ax.set_xticklabels([f"{e:g}" for e in EMIT_SMOOTHINGS], rotation=30)
        ax.set_xlabel("emit_smoothing")
        ax.set_ylabel("micro-F1")
        ax.set_title(lang)
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(f1):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    fig.suptitle("HMM: emit_smoothing vs validation micro-F1")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hmm_emit_smoothing_curves.png", dpi=150)
    plt.close(fig)

    # 热力图：行=语言，列=平滑
    heat = np.stack([results["English_f1"], results["Chinese_f1"]], axis=0)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    im = ax2.imshow(heat, aspect="auto", cmap="YlGn")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(LANGUAGES)
    ax2.set_xticks(range(len(EMIT_SMOOTHINGS)))
    ax2.set_xticklabels([f"{e:g}" for e in EMIT_SMOOTHINGS])
    ax2.set_xlabel("emit_smoothing")
    for i in range(2):
        for j in range(len(EMIT_SMOOTHINGS)):
            ax2.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig2.colorbar(im, ax=ax2, fraction=0.03, label="micro-F1")
    ax2.set_title("HMM validation micro-F1 heatmap")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "hmm_emit_smoothing_heatmap.png", dpi=150)
    plt.close(fig2)
    print(f"\n已保存: {FIG_DIR}/hmm_results.npz 与 PNG 图")


if __name__ == "__main__":
    main()
