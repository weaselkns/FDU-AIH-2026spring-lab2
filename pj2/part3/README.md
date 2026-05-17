# Part 3：Transformer + 手写线性链 CRF

- **`transformer_crf_ner.py`**：数据管线、Transformer、DDP 训练、维特比解码与写预测。
- **`linear_chain_crf.py`**：配分函数、NLL、解码（与作业要求一致的手写 CRF）。
- **`checkpoints/`**：提交用权重复本（`English_transformer_crf.pt`、`Chinese_transformer_crf.pt`）。

数据根目录为仓库下的 `NER/`（脚本从 `pj2/part3/` 向上解析到仓库根）。请在**仓库根目录**执行：

```bash
# 训练（示例）
torchrun --standalone --nproc_per_node=4 \
  pj2/part3/transformer_crf_ner.py \
  --lang both --batch-size 64 --num-workers 4 \
  --save-dir pj2/part3/checkpoints
```

**仅加载 checkpoint 解码**（须同时提供三个参数）：

```bash
python pj2/part3/transformer_crf_ner.py \
  --ckpt-path pj2/part3/checkpoints/English_transformer_crf.pt \
  --input-path NER/English/validation.txt \
  --output-path NER/predictions/transformer_crf/English_validation_transformer_crf.txt
```

（`python pj2/part3/transformer_crf_ner.py --help` 查看全部选项。）
