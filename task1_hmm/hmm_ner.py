"""
任务一：基于隐马尔可夫模型（HMM）的命名实体识别（NER）

从零实现 HMM 的「统计训练 + 维特比解码」，不依赖 sklearn / PyTorch 等机器学习框架。
仅使用 NumPy 做向量化与数值运算。

模型：
- 隐状态 y_t：第 t 个词的实体标签（BIO / BMESO 等）。
- 观测 x_t：第 t 个词（字/英文 token）。
- 一阶马尔可夫假设：P(y_t | y_{t-1}, ..., y_0) = P(y_t | y_{t-1})。
- 输出独立性假设：P(x_t | y_t, x_{t-1}, ...) = P(x_t | y_t)。

参数：
- 初始概率 π：句子第一个位置的标签分布 P(y_0)。
- 转移矩阵 A：P(y_t | y_{t-1})，行表示上一标签，列表示当前标签。
- 发射矩阵 B：P(x_t | y_t)，对「词表外」词统一走 UNK 发射列。

解码：维特比（Viterbi）在 log 域中求整条句子的最大概率标签路径，避免下溢。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 路径与随机种子
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    设置随机种子。
    参数：
        seed: 随机种子
    """
    np.random.seed(seed)


def project_root() -> Path:
    """返回仓库根目录（本文件位于 task1_hmm/ 下）。"""
    return Path(__file__).resolve().parent.parent


def ner_data_dir() -> Path:
    """课程提供的 NER 数据根目录。"""
    return project_root() / "NER"


# ---------------------------------------------------------------------------
# 数据读取：CoNLL 风格，词与标签空格分隔，句间空行
# ---------------------------------------------------------------------------

def read_tagged_corpus(path: str | Path) -> list[list[tuple[str, str]]]:
    """
    读取带 gold 标签的语料，按空行切分为句子。
    参数：
        path: train.txt 或 validation.txt 等文件路径
    Returns:
        sentences: 每个元素是一条句子的 (token, tag) 列表
    """
    path = Path(path)
    # sentences 的每个元素是一个句子，每个句子由多个 (token, tag) 组成，比如：sentence=[[("张", "B-PER"), ("三", "I-PER")], [("北", "B-LOC"), ("京", "I-LOC")]]
    sentences: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) < 2:
                # 极少数异常行：跳过，避免整程序崩溃
                continue
            token, tag = parts[0], parts[1]
            cur.append((token, tag))
    if cur:
        sentences.append(cur)
    return sentences


def read_tokens_only(path: str | Path) -> list[list[str]]:
    """
    读取仅用于预测的句子（每行若有两列则只取第一列 token）。
    参数：
        path: 与训练相同格式的文件（可为无标签测试集）
    Returns:
        每条句子为 token 列表
    """
    path = Path(path)
    # sents 的每个元素是一个句子，每个句子由多个 token 组成，比如：sents=[["张", "三"], ["北", "京"]]
    sents: list[list[str]] = []
    cur: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            token = line.split()[0]
            cur.append(token)
    if cur:
        sents.append(cur)
    return sents


def write_predictions(
    path: str | Path,
    sentences_tokens: list[list[str]],
    pred_tags: list[list[str]],
) -> None:
    """
    写出与 example_my_result.txt 一致的格式：每行「token pred_tag」，句间空行。
    参数：
        path: 输出路径
        sentences_tokens: 与 pred_tags 等长的分句 token 列表
        pred_tags: 与 sentences_tokens 等长的预测标签序列

    例子：
        输入：
            sentences_tokens = [["张", "三"], ["北", "京"]]
            pred_tags = [["B-PER", "I-PER"], ["B-LOC", "I-LOC"]]
        输出：
            张 B-PER
            三 I-PER

            北 B-LOC
            京 I-LOC
            
    """
    assert len(sentences_tokens) == len(pred_tags)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for toks, tags in zip(sentences_tokens, pred_tags):
            assert len(toks) == len(tags)
            for w, t in zip(toks, tags):
                f.write(f"{w} {t}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# HMM：极大似然计数 + 加性平滑；维特比解码
# ---------------------------------------------------------------------------

class HMMNER:
    """
    一阶 HMM 序列标注器。

    训练：在全部句子上统计
        - 句首标签频次 -> π
        - 相邻标签对频次 -> A
        - (词, 标签) 共现频次 -> B（含 UNK 列）

    平滑：对 π、A、B 的分母/分子加伪计数，避免 log(0)，并给未见词分配非零发射概率。
    """

    def __init__(
        self,
        trans_smoothing: float = 1e-3,
        emit_smoothing: float = 1e-3,
        init_smoothing: float = 1e-3,
    ) -> None:
        """
        参数：
            trans_smoothing: 转移矩阵 Dirichlet/Laplace 风格伪计数
            emit_smoothing: 发射矩阵伪计数（越大则越「均匀」，对稀疏更稳）
            init_smoothing: 初始分布伪计数
        """
        self.trans_smoothing = float(trans_smoothing)
        self.emit_smoothing = float(emit_smoothing)
        self.init_smoothing = float(init_smoothing)

        # 训练完成后赋值
        # 训练集里出现过的所有标签字符串
        self.tags: list[str] = []
        self.tag_to_idx: dict[str, int] = {}
        self.idx_to_tag: list[str] = []

        # 训练集里出现过的所有词型
        self.vocab: list[str] = []
        self.word_to_idx: dict[str, int] = {}
        self.unk_idx: int = -1

        # log 参数（维特比直接使用），形状标注中，S 表示标签数，V 表示词数（含 UNK 列）
        self.log_pi: np.ndarray | None = None  # (S,)
        self.log_A: np.ndarray | None = None  # (S, S)
        self.log_B: np.ndarray | None = None  # (S, V)  V 含 UNK

    def _build_vocab(self, sentences: list[list[tuple[str, str]]]) -> None:
        """
        根据训练语料构建词表，并追加特殊符号 __UNK__ 作为「集外词」桶。
        NER 中词表往往很大；集外词在测试句里很常见，必须单独一列发射。
        """
        # 遍历所有句子里每个 (w, _)，对词 w 计数
        freq: dict[str, int] = {}
        for sent in sentences:
            for w, _ in sent:
                freq[w] = freq.get(w, 0) + 1
        # 保留所有在训练中出现过的词型（也可改为去掉极低频以压缩词表）
        self.vocab = sorted(freq.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.unk_idx = len(self.vocab)
        self.vocab.append("__UNK__")

    def _build_tag_index(self, sentences: list[list[tuple[str, str]]]) -> None:
        """根据训练语料中出现的标签集合建立稳定下标（按字典序，便于复现）。"""
        # 遍历所有句子里每个 (_, y)，对标签 y 计数
        tag_set: set[str] = set()
        for sent in sentences:
            for _, y in sent:
                tag_set.add(y)
        self.tags = sorted(tag_set)
        self.tag_to_idx = {t: i for i, t in enumerate(self.tags)}
        self.idx_to_tag = self.tags.copy()

    def fit(self, sentences: list[list[tuple[str, str]]]) -> None:
        """
        在带标签的句子上估计 HMM 参数（极大似然 + 平滑）。
        参数：
            sentences: read_tagged_corpus 的输出
        输出：无返回值，但是在对象中写了：
            log_pi: (S,)，初始概率
            log_A: (S, S)，转移矩阵，a_ij = P(y_t=j | y_t-1=i)
            log_B: (S, V)，发射矩阵，b_yk = P(x=w_k | y)
        """
        if not sentences:
            raise ValueError("训练集为空，无法估计参数。")

        # 初始化
        self._build_tag_index(sentences)
        self._build_vocab(sentences)

        S = len(self.tags)
        V = len(self.vocab)  # 已含 UNK 列

        # 计数容器
        # pi 为句首标签频数
        # trans[i,:] 为上一标签是 i 时，转移到下一标签的联合频数
        # emit[i,:] 为标签是 i 时，各词型的频数
        pi = np.zeros(S, dtype=np.float64)
        trans = np.zeros((S, S), dtype=np.float64)
        emit = np.zeros((S, V), dtype=np.float64)

        for sent in sentences:
            if not sent:
                continue
            # 句首：对第一个标签计数
            # 比如：sent = [("张", "B-PER"), ("三", "I-PER")]，则 y0 = "B-PER"
            y0 = self.tag_to_idx[sent[0][1]]
            pi[y0] += 1.0

            # 发射：每个位置 (词 -> 列)，UNK 列在训练中不直接累加（保持 0，平滑后仍会有合理概率质量分给 UNK）
            # 含义是在标签 yi 下，词 wi 出现的次数
            # 比如：sent = [("张", "B-PER"), ("三", "I-PER")]，则 w = "张"，yi = 1，wi = 0
            for w, y in sent:
                yi = self.tag_to_idx[y]
                wi = self.word_to_idx.get(w, self.unk_idx)
                emit[yi, wi] += 1.0

            # 转移：相邻标签对
            # 含义是在标签 y_prev 下，标签 y_cur 出现的次数
            # 比如：sent = [("张", "B-PER"), ("三", "I-PER")]，则 y_prev = "B-PER"，y_cur = "I-PER"
            for (w_prev, y_prev), (w_cur, y_cur) in zip(sent, sent[1:]):
                i = self.tag_to_idx[y_prev]
                j = self.tag_to_idx[y_cur]
                trans[i, j] += 1.0

        eps_t = self.trans_smoothing
        eps_e = self.emit_smoothing
        eps_i = self.init_smoothing

        # 初始分布：加平滑后归一化，每行和为 1
        pi = pi + eps_i
        pi /= pi.sum()

        # 转移：对每一行（上一标签固定）在「下一标签」维度上归一化
        trans += eps_t
        row_sums = trans.sum(axis=1, keepdims=True)
        # 防止除零（理论上不会）
        row_sums = np.maximum(row_sums, 1e-300)
        trans = trans / row_sums

        # 发射：对每个标签 y，在所有词列（含 UNK）上归一化
        emit += eps_e
        emit /= emit.sum(axis=1, keepdims=True)

        self.log_pi = np.log(pi)
        self.log_A = np.log(trans)
        self.log_B = np.log(emit)

    def predict_sentence(self, tokens: list[str]) -> list[str]:
        """
        对单句做维特比解码，返回与 tokens 等长的标签序列。
        参数：
            tokens: 一条句子的词序列
        Returns:
            预测标签（字符串 BIO/BMES 等）
        维特比 = 从左读到右，每一步都记「到这儿为止最好的成绩」和「谁让我成绩最好」；读完最后一词挑冠军，再顺着「谁让我最好」往回走，就得到整句标签。
        """
        if self.log_pi is None or self.log_A is None or self.log_B is None:
            raise RuntimeError("请先调用 fit() 完成参数估计。")

        T = len(tokens)
        S = len(self.tags)
        if T == 0:
            return []

        # 词 -> 列下标；集外词走 UNK 列
        # obs 即 observation，obs[t] 表示第 t 个词 在发射矩阵 log_B 中的列下标
        obs = np.empty(T, dtype=np.int64)
        for t, w in enumerate(tokens):
            obs[t] = self.word_to_idx.get(w, self.unk_idx)

        # dp[t, j]: 第 t 个词落在标签 j 的最大 log 概率
        dp = np.full((T, S), -np.inf, dtype=np.float64)
        back = np.zeros((T, S), dtype=np.int64)

        # ---------- 初始化（t = 0）----------
        # δ_0(j) = log π_j + log b_j(x_0)；无上一状态，故不含转移项。
        dp[0] = self.log_pi + self.log_B[:, obs[0]]

        # ---------- 递推（t = 1 .. T-1）----------
        # δ_t(j) = max_i [ δ_{t-1}(i) + log a_{ij} ] + log b_j(x_t)
        # 向量化：prev[i, j] = dp[t-1, i] + logA[i, j]，对 j 在 i 维上取 argmax。
        for t in range(1, T):
            # prev 形状 (S, S)：prev[i, j] 表示「上一标签为 i、当前为 j」的路径得分
            prev = dp[t - 1][:, None] + self.log_A
            best_prev = np.argmax(prev, axis=0)  # 每个当前标签 j 对应的最佳上一标签 i*
            best_score = prev[best_prev, np.arange(S)]  # 取出对应的 max_i 数值
            dp[t] = best_score + self.log_B[:, obs[t]]
            back[t] = best_prev

        # ---------- 回溯 ----------
        # 从最后一个词的最佳标签开始，沿 back 指针还原整条路径。
        path_idx = np.empty(T, dtype=np.int64)
        path_idx[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 2, -1, -1):
            path_idx[t] = back[t + 1, path_idx[t + 1]]

        return [self.idx_to_tag[int(j)] for j in path_idx]

    def save(self, path: str | Path) -> None:
        """
        将已训练参数持久化为 .npz，便于面试/测试阶段直接加载解码。
        说明：不保存平滑超参；加载后仅用于 predict，无需再次 fit。
        参数：
            path: 输出文件路径，例如 English_hmm.npz
        """
        if self.log_pi is None:
            raise RuntimeError("模型尚未训练，无法保存。")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            tags=np.array(self.tags, dtype=object),
            vocab=np.array(self.vocab, dtype=object),
            log_pi=self.log_pi,
            log_A=self.log_A,
            log_B=self.log_B,
        )
        print(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "HMMNER":
        """
        从 save() 写出的 .npz 恢复模型（仅用于解码）。
        参数：
            path: .npz 路径
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        hmm = cls()
        hmm.tags = data["tags"].tolist()
        hmm.tag_to_idx = {t: i for i, t in enumerate(hmm.tags)}
        hmm.idx_to_tag = hmm.tags.copy()
        hmm.vocab = data["vocab"].tolist()
        hmm.word_to_idx = {w: i for i, w in enumerate(hmm.vocab)}
        hmm.unk_idx = len(hmm.vocab) - 1
        hmm.log_pi = np.asarray(data["log_pi"], dtype=np.float64)
        hmm.log_A = np.asarray(data["log_A"], dtype=np.float64)
        hmm.log_B = np.asarray(data["log_B"], dtype=np.float64)
        print(f"模型已加载: {path}")
        return hmm

    def predict_corpus(self, sentences: list[list[str]]) -> list[list[str]]:
        """
        对多条句子逐句维特比。
        参数：
            sentences: 分句的 token 列表
        Returns:
            与输入等长的预测标签列表的列表
        """
        return [self.predict_sentence(s) for s in sentences]


# ---------------------------------------------------------------------------
# 训练 + 验证集评测流程
# ---------------------------------------------------------------------------

def train_and_decode_validation(
    language: str,
    ner_root: Path | None = None,
    out_dir: Path | None = None,
    trans_smoothing: float = 1e-3,
    emit_smoothing: float = 1e-3,
    init_smoothing: float = 1e-3,
    save_model_path: Path | None = None,
) -> tuple[Path, Path, HMMNER]:
    """
    在指定语言的 train.txt 上估计 HMM，对 validation.txt 解码并写出预测文件。
    参数：
        language: "English" 或 "Chinese"
        ner_root: NER 数据根目录，默认仓库下 NER/
        out_dir: 预测文件输出目录，默认 NER/predictions/hmm/
    Returns:
        (预测文件路径, gold 验证集路径, 训练好的 HMM 对象)
    """
    ner_root = ner_root or ner_data_dir()
    out_dir = out_dir or (ner_root / "predictions" / "hmm")
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = ner_root / language
    train_path = sub / "train.txt"
    val_path = sub / "validation.txt"
    pred_path = out_dir / f"{language}_validation_hmm.txt"

    print(f"[{language}] 读取训练集: {train_path}")
    train_sents = read_tagged_corpus(train_path)  # 读训练
    print(f"[{language}] 句子数: {len(train_sents)}")

    print(f"[{language}] 估计 HMM 参数（极大似然 + 平滑）...")
    # 建模型并 fit
    hmm = HMMNER(
        trans_smoothing=trans_smoothing,
        emit_smoothing=emit_smoothing,
        init_smoothing=init_smoothing,
    )
    hmm.fit(train_sents)

    print(f"[{language}] 读取验证集 token: {val_path}")
    val_tokens = read_tokens_only(val_path)
    print(f"[{language}] 解码验证集（维特比）...")
    pred_tags = hmm.predict_corpus(val_tokens)

    print(f"[{language}] 写出预测: {pred_path}")
    write_predictions(pred_path, val_tokens, pred_tags)

    if save_model_path is not None:
        hmm.save(save_model_path)

    return pred_path, val_path, hmm


def print_check_hint(language: str, gold_path: Path, pred_path: Path) -> None:
    """打印如何用课程提供的 check.py 做 micro-F1 评测（需在 NER 目录下运行）。"""
    # 相对 NER 目录的路径写法
    gold_rel = os.path.relpath(gold_path, ner_data_dir())
    pred_rel = os.path.relpath(pred_path, ner_data_dir())
    print("\n--- 使用 NER/check.py 评测（示例）---")
    print("请先: cd NER")
    print(
        "再运行 Python 一行调用，例如:\n"
        "python -c \"from check import check; "
        f"check(language='{language}', gold_path=r'{gold_rel}', my_path=r'{pred_rel}')\""
    )


def main() -> None:
    """
    入口：对中文、英文（或命令行指定语言）训练 HMM 并在验证集上生成预测。
    亦支持「仅解码」：传入 --model-path、--input-path、--output-path，
    用于面试下发的 test.txt（格式与 validation 相同，可为仅 token）。
    """
    parser = argparse.ArgumentParser(description="任务一：手写 HMM 做 NER")
    parser.add_argument(
        "--lang",
        choices=["English", "Chinese", "both"],
        default="both",
        help="选择训练/解码的语言子集",
    )
    parser.add_argument(
        "--emit-smoothing",
        type=float,
        default=1e-3,
        help="发射矩阵平滑强度；语料更稀疏时可略调大（如 1e-2）",
    )
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="",
        help="若非空，则在对应目录下写入 {English|Chinese}_hmm.npz 便于复用",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="与 --input-path 联用：加载已训练 .npz，对测试文件维特比解码",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
        help="待标注文件路径（CoNLL 空行分句；每行至少一列为 token）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="写出预测结果的文件路径",
    )
    args = parser.parse_args()

    set_seed(2026)

    # 仅解码模式：加载模型 + 输入文件 -> 输出文件
    if args.model_path or args.input_path or args.output_path:
        if not (args.model_path and args.input_path and args.output_path):
            raise SystemExit(
                "仅解码模式需要同时提供 --model-path、--input-path、--output-path"
            )
        hmm = HMMNER.load(args.model_path)
        sents = read_tokens_only(args.input_path)
        tags = hmm.predict_corpus(sents)
        write_predictions(args.output_path, sents, tags)
        print(f"已写出预测: {args.output_path}")
        return

    langs = ["English", "Chinese"] if args.lang == "both" else [args.lang]
    save_dir = Path(args.save_model_dir) if args.save_model_dir else None

    for lang in langs:
        model_path = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = save_dir / f"{lang}_hmm.npz"
        pred_path, gold_path, _ = train_and_decode_validation(
            language=lang,
            emit_smoothing=args.emit_smoothing,
            save_model_path=model_path,
        )
        print_check_hint(lang, gold_path, pred_path)


if __name__ == "__main__":
    main()
