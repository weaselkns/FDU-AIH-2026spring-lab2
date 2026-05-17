# Project 2：命名实体识别（NER）

本仓库包含课程作业的三个实现部分：**HMM**、**CRF（sklearn-crfsuite）**、**Transformer + 手写 CRF**，以及课程提供的 `NER/` 数据与评测脚本。

---

## 一、环境准备

### 1. 建议 Python 版本

Python **3.10+**（与类型注解、`torch` 等兼容较好）。

### 2. 安装依赖

在**项目根目录** `pj2/` 下执行：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

| 包 | 用途 |
|----|------|
| `numpy` | 任务一 HMM |
| `scikit-learn` | `NER/check.py` 评测 |
| `sklearn-crfsuite` | 任务二 CRF |
| `torch` | 任务三 Transformer + 手写 CRF（多卡需 `torchrun`） |

### 3. 数据与目录约定

- 语料与标签说明位于 **`NER/English/`**、**`NER/Chinese/`**（`train.txt`、`validation.txt`、`tag.txt`）。
- 所有脚本默认从**仓库根目录**解析路径（即 `pj2/` 下应能看到 `NER/`、`task1_hmm/` 等文件夹）。
- **运行命令前请先 `cd` 到项目根目录**（下文 `<REPO_ROOT>` 表示该路径）。

---

## 二、任务一：手写 HMM（`task1_hmm/hmm_ner.py`）

### 作用

在训练集上统计 HMM 参数（初始、转移、发射 + 平滑），对验证集做 **维特比解码**，生成与 `example_data/example_my_result.txt` 相同格式的预测文件。

### 常用命令

**中英文一起训练并解码验证集（默认）：**

```bash
cd <REPO_ROOT>
python task1_hmm/hmm_ner.py
```

**只跑英文或中文：**

```bash
python task1_hmm/hmm_ner.py --lang English
python task1_hmm/hmm_ner.py --lang Chinese
```

**保存模型（面试 `test.txt` 可再加载解码）：**

```bash
python task1_hmm/hmm_ner.py --lang both --save-model-dir task1_hmm/models
```

**仅加载模型解码（需先训练并保存过 `.npz`）：**

```bash
python task1_hmm/hmm_ner.py ^
  --model-path task1_hmm/models/English_hmm.npz ^
  --input-path NER/English/validation.txt ^
  --output-path NER/predictions/hmm/my_english_result.txt
```

（Linux/macOS 将 `^` 换为行末 `\`。）

### 输出位置

| 语言 | 默认预测文件 |
|------|----------------|
| English | `NER/predictions/hmm/English_validation_hmm.txt` |
| Chinese | `NER/predictions/hmm/Chinese_validation_hmm.txt` |

### 评测（micro-F1）

```bash
cd NER
python -c "from check import check; check(language='English', gold_path=r'English/validation.txt', my_path=r'predictions/hmm/English_validation_hmm.txt')"
python -c "from check import check; check(language='Chinese', gold_path=r'Chinese/validation.txt', my_path=r'predictions/hmm/Chinese_validation_hmm.txt')"
```

---

## 三、任务二：CRF（`task2_crf/crf_ner.py`）

### 作用

用手工特征模板 + **sklearn-crfsuite** 训练线性链 CRF，对验证集维特比解码。**训练在 CPU 上**，与 GPU 数量无关。

### 常用命令

**中英文默认都跑：**

```bash
cd <REPO_ROOT>
python task2_crf/crf_ner.py
```

**调正则与 L-BFGS 迭代次数：**

```bash
python task2_crf/crf_ner.py --lang English --c1 0.1 --c2 0.5 --max-iterations 100
```

**保存 / 加载模型（`pickle`）：**

```bash
python task2_crf/crf_ner.py --lang Chinese --save-model-dir task2_crf/models
python task2_crf/crf_ner.py --model-path task2_crf/models/Chinese_crf.pkl --input-path NER/Chinese/validation.txt --output-path NER/predictions/crf/my_chinese_crf.txt
```

### 输出位置

- `NER/predictions/crf/English_validation_crf.txt`
- `NER/predictions/crf/Chinese_validation_crf.txt`

### 评测

```bash
cd NER
python -c "from check import check; check(language='English', gold_path=r'English/validation.txt', my_path=r'predictions/crf/English_validation_crf.txt')"
python -c "from check import check; check(language='Chinese', gold_path=r'Chinese/validation.txt', my_path=r'predictions/crf/Chinese_validation_crf.txt')"
```

---

## 四、任务三：Transformer + 手写 CRF（`task3_transformer_crf/`）

### 文件说明

| 文件 | 说明 |
|------|------|
| `transformer_crf_ner.py` | 数据管线、训练、验证解码、可选 **DDP 多卡** |
| `linear_chain_crf.py` | **手写**线性链 CRF（配分函数、NLL、维特比） |

### 单机单卡（或自动使用一张 GPU）

```bash
cd <REPO_ROOT>
python task3_transformer_crf/transformer_crf_ner.py --lang both --epochs 12 --batch-size 32 --save-dir task3_transformer_crf/checkpoints
```

### 多卡分布式（DDP，推荐 Linux + NVIDIA）

**`--batch-size` 表示每张 GPU 上的 batch；** 全局等效 batch 约为 **`batch_size × GPU 数`**。

```bash
cd <REPO_ROOT>
torchrun --standalone --nproc_per_node=4 \
  task3_transformer_crf/transformer_crf_ner.py \
  --lang English --epochs 12 --batch-size 64 \
  --save-dir task3_transformer_crf/checkpoints --num-workers 2
```

仅 **rank 0** 会写验证集预测与 checkpoint，避免多进程抢写同一文件。

### 仅解码（加载 `.pt`）

```bash
python task3_transformer_crf/transformer_crf_ner.py \
  --ckpt-path task3_transformer_crf/checkpoints/English_transformer_crf.pt \
  --input-path NER/English/validation.txt \
  --output-path NER/predictions/transformer_crf/my_eng.txt
```

### 输出位置

- `NER/predictions/transformer_crf/English_validation_transformer_crf.txt`
- `NER/predictions/transformer_crf/Chinese_validation_transformer_crf.txt`

### 评测

```bash
cd NER
python -c "from check import check; check(language='English', gold_path=r'English/validation.txt', my_path=r'predictions/transformer_crf/English_validation_transformer_crf.txt')"
```

---

## 五、完整示例（从零到评测）

在仓库根目录依次执行（可按需跳过已跑过的任务）：

```bash
cd <REPO_ROOT>
pip install -r requirements.txt

# 任务一
python task1_hmm/hmm_ner.py --lang both

# 任务二
python task2_crf/crf_ner.py --lang both

# 任务三（单机单卡；有 GPU 会自动用 cuda）
python task3_transformer_crf/transformer_crf_ner.py --lang both --epochs 8 --batch-size 32 --save-dir task3_transformer_crf/checkpoints

# 评测（在 NER 目录下）
cd NER
python -c "from check import check; check(language='English', gold_path=r'English/validation.txt', my_path=r'predictions/hmm/English_validation_hmm.txt')"
python -c "from check import check; check(language='English', gold_path=r'English/validation.txt', my_path=r'predictions/crf/English_validation_crf.txt')"
python -c "from check import check; check(language='English', gold_path=r'English/validation.txt', my_path=r'predictions/transformer_crf/English_validation_transformer_crf.txt')"
```

中文把上面三条里的 `English` 换成 `Chinese`、路径换成 `Chinese/...` 与对应 `predictions/...` 即可。

---

## 六、GitHub → 服务器 → 回传本地

### 1. 本地：提交并推送

确保 `NER/` 下**训练所需**的 `train.txt`、`validation.txt` 等已纳入版本控制（或按课程要求单独上传数据，见下文「数据说明」）。

```bash
cd <REPO_ROOT>
git status
git add .
git commit -m "feat: NER task1-3 and README"
git remote add origin https://github.com/<你的用户名>/<仓库名>.git   # 若尚未添加
git push -u origin main
```

### 2. 服务器：克隆与运行

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
pip install -r requirements.txt
# 然后按第二节～四节运行各任务；任务三多卡见第四节 torchrun 示例
```

### 3. 回传结果到本地（推荐 `scp` / `rsync`）

**预测与 checkpoint 默认被 `.gitignore` 忽略**，一般不通过 `git push` 传大文件；用拷贝更干净。

在**本地终端**执行（按你的服务器地址与路径修改）：

```bash
# 示例：把整个 predictions 与 checkpoints 拉回本地仓库对应位置
scp -r user@server.example.com:~/pj2/NER/predictions ./NER/
scp -r user@server.example.com:~/pj2/task3_transformer_crf/checkpoints ./task3_transformer_crf/
```

或使用 `rsync`（可断点续传、排除无关文件）：

```bash
rsync -avz user@server.example.com:~/pj2/NER/predictions/ ./NER/predictions/
rsync -avz user@server.example.com:~/pj2/task1_hmm/models/ ./task1_hmm/models/
```

若你希望**预测文件也进 Git**，可从 `.gitignore` 中删掉对应规则后再 `git add`，注意仓库体积与隐私。

### 4. 数据说明

- 若 **GitHub 不包含** `NER/Chinese`、`NER/English`（体积策略），服务器克隆后需从课程网盘 / 本地拷贝 **`NER` 数据目录** 到仓库内同名路径，再运行脚本。
- `NER.rar` 已在 `.gitignore` 中忽略；解压后的 `NER/` 若已提交则服务器 `git clone` 即可。

### 5. `.gitignore` 生效说明

若你**曾经**把 `NER/predictions/` 等目录提交进 Git，后来才加入 `.gitignore`，Git **仍会继续跟踪**这些文件。需要停止跟踪时（保留本地文件）：

```bash
git rm -r --cached NER/predictions/
git commit -m "chore: stop tracking generated predictions"
```

### 6. Windows 与 UTF-8

`NER/check.py` 已使用 `encoding="utf-8"` 打开文件；若在 Windows 下直接双击或旧终端乱码，请在 **UTF-8 终端**（如 Windows Terminal + PowerShell 7）中运行上述命令。

---

## 七、学术诚信

作业要求独立完成、面试会问代码与原理；请勿抄袭，提交前确保自己理解各文件职责与超参含义。

---

## 八、目录结构（核心）

```
pj2/
├── README.md                 # 本说明
├── requirements.txt
├── NER/                      # 数据与 check.py
│   ├── check.py
│   ├── English/ , Chinese/
│   └── predictions/          # 运行后生成（默认不提交）
├── task1_hmm/hmm_ner.py
├── task2_crf/crf_ner.py
└── task3_transformer_crf/
    ├── linear_chain_crf.py
    └── transformer_crf_ner.py
```

（`pj1/` 若仅作参考可保持 `.gitignore` 忽略。）
