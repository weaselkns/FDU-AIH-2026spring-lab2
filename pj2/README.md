# PJ2 提交目录说明

本目录为 **Project 2：NER** 的实验报告与**自包含实现代码**（Part 1/2/3 在 `part*/`，Bonus 在 `bonus/`），便于单独打包提交。

## 目录结构

| 路径 | 内容 |
|------|------|
| `report.md` | 实验报告（正文中的图路径相对于本目录） |
| `requirements.txt` | Python 依赖（含 Part 3 的 `tqdm`） |
| `_eval_utils.py` | Part 1/2 批量实验用的 micro-F1 计算 |
| `part1/hmm_ner.py` | 手写 HMM 实现 |
| `part1/hmm_experiments.py` | HMM 发射平滑扫描 + 出图 |
| `part1/figures/` | HMM 曲线图、热力图（`.png`） |
| `part1/results/` | `hmm_results.npz`（扫描得到的 micro-F1 等） |
| `part2/crf_ner.py` | sklearn-crfsuite 线性链 CRF |
| `part2/crf_experiments.py` | c1/c2 扫描 + 出图 |
| `part2/figures/` | CRF 曲线与柱状图（`.png`） |
| `part2/results/` | `crf_results.npz` |
| `part3/transformer_crf_ner.py` | Transformer + 手写 CRF 训练/解码 |
| `part3/linear_chain_crf.py` | 手写线性链 CRF |
| `part3/checkpoints/` | 训练得到的 `.pt` 权重复本 |
| `part3/README.md` | Part 3 解码命令示例 |
| `bonus/` | Bonus：`template_for_crf.utf8` + 模板特征解析 + 中文 CRF 训练脚本（见 `bonus/README.md`） |

## 运行前提

- 课程数据 **`NER/`** 须位于本目录的**上一级**（即仓库根目录与 `pj2/` 同级），路径形如 `<REPO_ROOT>/NER/English/...`。
- 在 **`<REPO_ROOT>`** 下执行下列命令（不要把 `cd` 进 `pj2` 再跑训练脚本，否则找不到 `NER/`）。

## 复现实验（在仓库根目录）

```bash
# Part 1 批量实验（会写回 part1/figures/ 与 part1/results/）
python pj2/part1/hmm_experiments.py

# Part 2 批量实验（会写回 part2/figures/ 与 part2/results/）
python pj2/part2/crf_experiments.py

# Part 3 多卡训练（示例）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  pj2/part3/transformer_crf_ner.py \
  --lang both --batch-size 64 --num-workers 4 \
  --save-dir pj2/part3/checkpoints

# Bonus（中文模板 CRF）
python pj2/bonus/train_template_crf_chinese.py
```

评测仍在 **`NER/`** 目录下用课程提供的 `check.py`（见 `report.md`）。
