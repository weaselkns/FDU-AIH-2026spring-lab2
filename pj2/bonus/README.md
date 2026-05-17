# Bonus：CRFsuite 模板特征 + 中文 NER

作业要求：理解课堂 CRF 分词特征形式，将给定 **`template_for_crf.utf8`** 用于 NER。

## 文件

| 文件 | 作用 |
|------|------|
| `template_for_crf.utf8` | 与 `NER/template_for_crf.utf8` 一致（U00–U09、B00–B09，列 0 为当前字） |
| `template_crf_features.py` | 解析模板并按字窗实例化特征字典 |
| `train_template_crf_chinese.py` | 读中文训练集 → 提模板特征 → `sklearn-crfsuite` L-BFGS 训练 → 验证集维特比解码 |

## 运行（仓库根目录）

```bash
python pj2/bonus/train_template_crf_chinese.py
```

默认与 Part 2 相同 `c1=0.08`、`c2=0.4`。预测写入  
`NER/predictions/bonus_template_crf/Chinese_validation_bonus_template_crf.txt`。

与 **Part 2** 的差异：Part 2 中文用手工特征键（`c`、`p1+n1` 等）；Bonus **仅**使用模板定义的 20 组字窗/字组观测，更贴近 CRFsuite 分词教程中的特征表。
