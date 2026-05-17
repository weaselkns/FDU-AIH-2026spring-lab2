# Artificial Intelligence(H) PJ2 实验报告

孔恩燊　23307130021　2026.5

本报告对应课程 **Project 2：命名实体识别（NER）**，实现并评测三条序列标注路线：**手写 HMM**、**线性链 CRF（sklearn-crfsuite + 手工特征）**、**Transformer 编码器 + 手写线性链 CRF**。数据为课程提供的 `NER/English` 与 `NER/Chinese`；指标统一为 `NER/check.py` 输出的 **实体级 micro-F1**（`labels` 不含 `O`）。

批量对比实验脚本与图保存在 `pj2/part1/`、`pj2/part2/`；**Part 3 暂不跑超参网格**，仅记录代表性训练配置与验证集结果。

---

## Part 1：手写 HMM 实现 NER

### 1.1 任务与数据

Part 1 要求**不使用机器学习框架**，手写隐马尔可夫模型完成 NER。数据为 CoNLL 风格：`train.txt` / `validation.txt` 中非空行为 `token tag`，句间空行分句。

- **English**：BIO 标注，9 类标签（含 `O`），训练约 14041 句。
- **Chinese**：BMESO 标注，33 类标签（含 `O`），训练约 3820 句。

模型假设一阶马尔可夫：\(P(\mathbf{y}\mid\mathbf{x}) \propto \pi_{y_1}\prod_t P(y_t\mid y_{t-1})\,P(x_t\mid y_t)\)。在训练集上统计**初始分布 \(\pi\)**、**转移矩阵 \(A\)**、**发射矩阵 \(B\)**（词表外统一映射为 `UNK`），并对计数做 **Laplace 风格平滑** 避免零概率。解码在 **log 域** 做维特比，求最大后验标签路径。

### 1.2 实现要点

代码位于 `task1_hmm/hmm_ner.py`，核心类为 `HMMNER`：

| 函数 / 方法 | 作用 |
|------|------|
| `read_tagged_corpus` / `read_tokens_only` | 读入带标签语料或仅 token 的待解码文件 |
| `HMMNER.fit` | 统计 \(\pi, A, B\) 并平滑 |
| `HMMNER.predict_sentence` | 单句维特比解码 |
| `train_and_decode_validation` | 训练 → 验证集解码 → 写预测文件 |
| `HMMNER.save` / `load` | 面试用 `test.txt` 可加载 `.npz` 再解码 |

默认平滑：`trans_smoothing = emit_smoothing = init_smoothing = 1e-3`；命令行可通过 `--emit-smoothing` 调节发射平滑（转移/初始在函数参数中亦可传入）。

### 1.3 对比实验设置

在 `pj2/part1/hmm_experiments.py` 中，固定 `trans_smoothing = init_smoothing = 1e-3`，对 **emit_smoothing** 做网格搜索：

```python
EMIT_SMOOTHINGS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
LANGUAGES = ["English", "Chinese"]
```

每种配置在对应语言训练集上重新估计 HMM，对验证集维特比解码，并用与 `check.py` 一致的标签集合计算 **micro-F1**。结果写入 `pj2/part1/figures/hmm_results.npz`。

### 1.4 实验结果与分析

下图给出不同发射平滑强度下的验证 micro-F1 曲线：

![HMM emit_smoothing 曲线](part1/figures/hmm_emit_smoothing_curves.png)

热力图汇总如下：

![HMM micro-F1 热力图](part1/figures/hmm_emit_smoothing_heatmap.png)

根据本次 `hmm_results.npz`：

| 语言 | 最佳 emit_smoothing | 验证 micro-F1 |
|------|---------------------|---------------|
| English | **0.01** | **0.789** |
| Chinese | **0.1** | **0.889** |

**分析**：

- 发射平滑过小（\(10^{-4}\sim10^{-3}\)）时，稀疏词的发射估计方差大，英文 F1 约 0.74～0.75；适度增大平滑（英文 \(10^{-2}\)、中文 \(10^{-1}\)）可缓解未登录词过拟合，F1 明显提升。
- 平滑过大（0.1 对英文）会削弱判别性，英文 F1 略回落至 0.781。
- 中文语料规模较小、标签 scheme 更细，略大的发射平滑整体更稳。
- HMM 作为**生成式**模型，对英文长句与 O 标签占比较高（约 83%）的分布仍能取得约 **0.79** 的实体 F1，但低于后续判别式 CRF / 神经网络方法，符合理论预期。

---

## Part 2：线性链 CRF 实现 NER

### 2.1 任务与模型

Part 2 使用 **sklearn-crfsuite** 训练**判别式**线性链 CRF：在手工特征上直接建模 \(P(\mathbf{y}\mid\mathbf{x})\)，不要求观测条件独立。势函数包含一元特征（字/词、小写、词形、标点、数字等）与二元转移特征；优化为 **L-BFGS**，带 **L1（c1）**、**L2（c2）** 正则。解码仍为维特比，与 HMM 动态规划形式相同，但分数来自特征权重而非生成概率。

实现见 `task2_crf/crf_ner.py`：

| 模块 | 作用 |
|------|------|
| `sentence_to_features_english` / `sentence_to_features_chinese` | 英文 / 中文特征模板 |
| `build_xy` | 构造 `X_train, y_train` |
| `make_crf` | 创建 `CRF(algorithm='lbfgs', c1, c2, ...)` |
| `train_and_decode_validation` | 训练、验证解码、写预测 |

默认超参：`c1=0.08`，`c2=0.4`，`max_iterations=80`。

### 2.2 对比实验设置

`pj2/part2/crf_experiments.py` 在固定 `max_iterations=80` 下做两组一维扫描（中英文各跑一遍）：

```python
C1_LIST = [0.0, 0.05, 0.08, 0.12, 0.2]   # 固定 c2=0.4
C2_LIST = [0.1, 0.2, 0.4, 0.8, 1.2]      # 固定 c1=0.08
```

结果保存为 `pj2/part2/figures/crf_results.npz`。

### 2.3 实验结果与分析

**L1（c1）扫描**（`c2=0.4`）：

![CRF c1 曲线](part2/figures/crf_c1_curves.png)

**L2（c2）扫描**（`c1=0.08`）：

![CRF c2 曲线](part2/figures/crf_c2_curves.png)

**各语言在两次扫描中的最优 F1**：

![CRF 最优 F1 柱状图](part2/figures/crf_best_f1_bar.png)

根据 `crf_results.npz` 汇总：

| 语言 | c1 扫描最优 | c2 扫描最优 | 备注 |
|------|-------------|-------------|------|
| English | 0.886（c1=0.08） | **0.889**（**c2=0.1**） | 略强于默认 c2=0.4 |
| Chinese | **0.954**（c1=0.05） | **0.954**（c2=0.2） | 整体高于英文 |

**分析**：

- 英文语料大、特征多：适度 L1（c1≈0.05～0.12）与较小 L2（c2≈0.1～0.4）可在拟合与泛化间折中；c2 过大（1.2）时 F1 降至约 0.877，正则过强。
- 中文验证集上 CRF 可达 **0.95** 左右 micro-F1，说明 BMES 模板 + 判别式建模对该数据集非常有效。
- 相较 Part 1 HMM，CRF 在英文上约提升 **10** 个百分点、中文上约 **6** 个百分点，体现**特征工程 + 判别式训练**的优势。

---

## Part 3：Transformer + 手写 CRF（无批量超参实验）

### 3.1 任务与结构

Part 3 要求 **Transformer 部分可用 PyTorch**，**CRF 部分必须手写**。本仓库实现为：

\[
\text{Embedding} \rightarrow \text{PositionalEncoding} \rightarrow \text{TransformerEncoder} \rightarrow \text{Linear} \rightarrow \text{LinearChainCRF}
\]

- **发射分数**：`proj(h_t)` 作为 CRF 一元势 \( \text{emit}_t(y_t) \)。
- **CRF**（`task3_transformer_crf/linear_chain_crf.py`）：可学习 `start_trans`、`end_trans`、`trans_matrix`；`batch_neg_log_likelihood` 用 log 域前向算法算 \(\log Z(x)\)；`batch_decode` 维特比解码。
- 训练：AdamW + 梯度裁剪；支持单卡与 `torchrun` **DDP** 多卡。

**说明**：Part 3 **未**像 Part 1/2 一样跑批量网格；下列结果为一次（或少数几次）代表性配置，供实验文档与面试说明。若需系统扫 `batch_size` / `lr` / `d_model`，可在 `transformer_crf_ner.py` 外包循环，代价为 GPU 时间显著增加。

### 3.2 默认超参（代码内，2026.5 更新）

| 项 | 默认值 |
|----|--------|
| `d_model` | 256 |
| `num_layers` | 4 |
| `nhead` | 8 |
| `dim_feedforward` | 1024（\(4\times d\_model\)） |
| `epochs` | 16 |
| `batch_size` | 64（**每 GPU**；DDP 时全局约 `64×卡数`） |
| `lr` | 1.5e-3 |
| `max_len` | 256 |
| `num_workers` | 4 |

运行示例见仓库根目录 `run.md`（须在项目根目录执行，勿在 `NER/` 子目录下相对路径调用）。

### 3.3 实现说明（主要文件）

| 文件 | 作用 |
|------|------|
| `transformer_crf_ner.py` | 数据管线、`TransformerCRFNER`、训练/解码、DDP |
| `linear_chain_crf.py` | 手写 CRF：NLL、维特比 |
| `NER/check.py` | 官方 micro-F1 评测 |

### 3.4 代表性训练现象与验证结果

**训练现象（经验）**：

- 有效 batch 过大（如 DDP 四卡 × 每卡 256）时，每 epoch 更新步数过少，英文易出现**高 precision、低 recall** 或标签分布异常；将 per-GPU `batch_size` 降至 **64**、或单卡训练，F1 明显回升。
- 中文数据量小，DDP 下每 epoch 仅数个 batch，训练 loss（NLL）数值可仍较高，但 **F1 可与 loss 数值不成比例地较好**（标签集更大、NLL 量级不同）。
- 小模型（`d_model=128`, 2 层）在调通数据与 batch 后，中文验证 micro-F1 可达 **0.83+**；增大至默认 256/4 层旨在进一步提升英文 recall，需重新训练后评测。

**一次较优配置下的验证 micro-F1**（`NER/predictions/transformer_crf/`，`check.py`）：

| 语言 | micro-F1 | 现象简述 |
|------|----------|----------|
| English | **≈ 0.52** | precision 高、recall 偏低（约 0.37），保守预测 |
| Chinese | **≈ 0.83** | precision / recall 较均衡 |

复现评测：

```bash
cd NER
python -c "from check import check; check(language='English', gold_path=r'English/validation.txt', my_path=r'predictions/transformer_crf/English_validation_transformer_crf.txt')"
python -c "from check import check; check(language='Chinese', gold_path=r'Chinese/validation.txt', my_path=r'predictions/transformer_crf/Chinese_validation_transformer_crf.txt')"
```

**分析**：

- 手写 CRF 层保证标签转移约束，与纯 softmax 逐 token 分类相比，在中文 BMES 上仍具优势。
- 未使用 BERT 等预训练编码器，英文词表大、上下文建模能力弱于 Part 2 的丰富离散特征，故英文 F1 可能**低于 CRF**；符合课程「可选预训练」的定位。
- 后续改进方向：英文单独训练、减小有效 batch、增加 epoch；或接入预训练 Transformer 冻结底层；Part 3 批量实验留作扩展。

---

## 三部分对比与总结

| 方法 | English micro-F1（验证集） | Chinese micro-F1（验证集） | 训练开销 |
|------|---------------------------|---------------------------|----------|
| HMM（Part 1，调平滑） | ≈ 0.79 | ≈ 0.89 | 极低，CPU |
| CRF（Part 2，调 c1/c2） | ≈ **0.89** | ≈ **0.95** | 中，CPU |
| Transformer+CRF（Part 3，代表运行） | ≈ 0.52 | ≈ 0.83 | 高，GPU |

**结论**：

1. 在本课程数据上，**手工特征的线性链 CRF** 仍是非常强的基线，尤其中文。
2. **HMM** 实现简单、可解释，适合作为生成式基线与面试维特比推导说明。
3. **Transformer+手写 CRF** 体现端到端神经网络与结构化预测结合；当前英文表现主要受**训练动态（batch/步数）与模型容量**影响，而非 CRF 实现错误。Part 1/2 的批量实验已用曲线与热力图给出超参敏感性；Part 3 以单次配置 + 现象分析为主，与「先不跑批量」的实验计划一致。

---

## 附录：复现实验命令

```bash
cd <REPO_ROOT>
pip install -r requirements.txt
pip install matplotlib   # 批量实验画图

# Part 1 批量实验
python pj2/part1/hmm_experiments.py

# Part 2 批量实验（英文较慢，建议 patience）
python pj2/part2/crf_experiments.py

# Part 3 训练（项目根目录）
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node=4 \
  task3_transformer_crf/transformer_crf_ner.py \
  --save-dir task3_transformer_crf/checkpoints
```

评测统一在 `NER/` 目录调用 `check.py`（见各部分正文示例）。
