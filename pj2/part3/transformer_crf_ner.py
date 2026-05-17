"""
任务三：Transformer 编码器 + 手写线性链 CRF 的命名实体识别

---------------------------------------------------------------------------
模型结构
---------------------------------------------------------------------------
1. Transformer 部分：PyTorch nn.TransformerEncoder（多头自注意力 + 前馈层），
   将离散 token 序列编码为上下文向量，再经线性层得到每个位置、每个标签的发射分数（作为 CRF 的一元势 / unary potential）。
2. CRF 部分：手写于 linear_chain_crf.py（配分函数、NLL、维特比），
   本文件负责数据管线、多卡训练、与 Transformer 组装成 nn.Module。

---------------------------------------------------------------------------
多卡分布式训练（DDP）
---------------------------------------------------------------------------
使用 PyTorch DistributedDataParallel（DDP）：每张 GPU 上跑一份进程，各进程处理数据的一个子集，
反向时在进程间同步梯度，等价于更大 batch 的随机梯度估计（--batch-size 表示每张卡上的 batch，全局约等于 batch_size × GPU 数）。

启动方式:
cd <项目根目录>
torchrun --standalone --nproc_per_node=4 \\
    pj2/part3/transformer_crf_ner.py \\
    --lang both --save-dir pj2/part3/checkpoints

单机单卡仍可直接 python ...（不初始化进程组）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# 支持 python pj2/part3/transformer_crf_ner.py 直接运行时的同目录导入
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sklearn import metrics

from linear_chain_crf import LinearChainCRF


# ---------------------------------------------------------------------------
# 分布式（DDP）辅助
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    """
    根据环境变量判断是否启用多进程分布式训练。

    torchrun / torch.distributed.launch 会注入: WORLD_SIZE, RANK, LOCAL_RANK

    Returns:
        use_ddp: 是否已初始化进程组且 world_size > 1
        local_rank: 本进程绑定的 GPU 序号（0 .. n-1）
        rank: 全局进程号（0 .. WORLD_SIZE-1）
        world_size: 进程总数
        device: 本进程应使用的 ``torch.device``
    """
    if not torch.cuda.is_available():
        return False, 0, 0, 1, torch.device("cpu")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        # 普通 python 单进程：不建进程组，默认用 cuda:0
        return False, 0, 0, 1, torch.device("cuda:0")

    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "检测到 WORLD_SIZE>1 但未设置 RANK/LOCAL_RANK。"
            "多卡请使用: torchrun --standalone --nproc_per_node=<GPU数> ..."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    # GPU 多卡优先 nccl；无 CUDA 时（极少用于多卡）退回 gloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True, local_rank, rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed(use_ddp: bool) -> None:
    """训练正常结束或异常时释放进程组（仅多卡 DDP 需要）。"""
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    DDP 包装后的 module 在 state_dict / 自定义方法（如 decode）上需访问内部裸模型：model.module；
    单卡时直接返回自身。
    """
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """仓库根目录（本文件位于 pj2/part3/）。"""
    return Path(__file__).resolve().parent.parent.parent


def ner_data_dir() -> Path:
    return project_root() / "NER"


def read_tagged_corpus(path: str | Path) -> list[list[tuple[str, str]]]:
    path = Path(path)
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            cur.append((parts[0], parts[1]))
    if cur:
        sents.append(cur)
    return sents


def read_tokens_only(path: str | Path) -> list[list[str]]:
    path = Path(path)
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
            cur.append(line.split()[0])
    if cur:
        sents.append(cur)
    return sents


def write_predictions(
    path: str | Path,
    sentences_tokens: list[list[str]],
    pred_tags: list[list[str]],
) -> None:
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
# 词表与标签表
# ---------------------------------------------------------------------------

PAD, UNK = "<pad>", "<unk>"

SORTED_LABELS_ENG = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]
SORTED_LABELS_CHN = [
    "O", "B-NAME", "M-NAME", "E-NAME", "S-NAME", "B-CONT", "M-CONT", "E-CONT", "S-CONT",
    "B-EDU", "M-EDU", "E-EDU", "S-EDU", "B-TITLE", "M-TITLE", "E-TITLE", "S-TITLE",
    "B-ORG", "M-ORG", "E-ORG", "S-ORG", "B-RACE", "M-RACE", "E-RACE", "S-RACE",
    "B-PRO", "M-PRO", "E-PRO", "S-PRO", "B-LOC", "M-LOC", "E-LOC", "S-LOC",
]


def normalize_token(word: str, language: str) -> str:
    """英文转小写，缓解词表稀疏与 UNK；中文保持原样。"""
    if language == "English":
        return word.lower()
    return word


def build_vocab_and_tags(
    sentences: list[list[tuple[str, str]]],
    language: str,
) -> tuple[dict[str, int], list[str], dict[str, int], list[str]]:
    """
    从训练句构造 token->id（0 保留给 PAD，UNK 固定为 1）与 tag->id。
    tag 列表按字典序稳定排列（与 HMM 一致）。
    """
    tok_set: set[str] = set()
    tag_set: set[str] = set()
    for s in sentences:
        for w, t in s:
            tok_set.add(normalize_token(w, language))
            tag_set.add(t)
    tag_itos = sorted(tag_set)
    tag_stoi = {t: i for i, t in enumerate(tag_itos)}

    # id 0: PAD；id 1: UNK；其余词按字典序编号（稳定、可复现）
    tok_itos = [PAD, UNK] + sorted(tok_set)
    tok_stoi = {w: i for i, w in enumerate(tok_itos)}
    return tok_stoi, tok_itos, tag_stoi, tag_itos


def _tag_entity_type(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag


def build_bio_transition_allow(tag_itos: list[str]) -> torch.Tensor:
    """英文 BIO：禁止 O→I-X、I-X→I-Y(X≠Y) 等非法转移。"""
    c = len(tag_itos)
    allow = torch.zeros(c, c, dtype=torch.bool)
    for i, ti in enumerate(tag_itos):
        for j, tj in enumerate(tag_itos):
            if ti == "O":
                allow[i, j] = tj == "O" or tj.startswith("B-")
            elif ti.startswith("B-") or ti.startswith("I-"):
                ei = _tag_entity_type(ti)
                if tj == "O" or tj.startswith("B-"):
                    allow[i, j] = True
                elif tj.startswith("I-"):
                    allow[i, j] = _tag_entity_type(tj) == ei
                else:
                    allow[i, j] = False
            else:
                allow[i, j] = True
    return allow


def use_bio_transition_mask(language: str, tag_itos: list[str]) -> bool:
    return language == "English" and "O" in tag_itos and any(t.startswith("B-") for t in tag_itos)


def compute_tag_weights(
    sentences: list[list[tuple[str, str]]],
    tag_itos: list[str],
    o_label: str = "O",
) -> torch.Tensor:
    """实体标签加权，缓解 O 主导与 MISC 等少数类。"""
    cnt: Counter[str] = Counter()
    for s in sentences:
        for _, t in s:
            cnt[t] += 1
    total = float(sum(cnt.values()) or 1.0)
    weights: list[float] = []
    for t in tag_itos:
        n = float(cnt.get(t, 0))
        if t == o_label:
            weights.append(1.0)
        else:
            weights.append(min(4.0, math.sqrt(total / (n + 1.0))))
    return torch.tensor(weights, dtype=torch.float32)


def micro_f1_from_tag_lists(
    language: str,
    y_true: list[str],
    y_pred: list[str],
) -> float:
    labels = (SORTED_LABELS_ENG if language == "English" else SORTED_LABELS_CHN)[1:]
    return float(
        metrics.f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    )


# ---------------------------------------------------------------------------
# 位置编码（Transformer 正弦位置编码）
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    正弦位置编码。
    与可学习位置嵌入相比：外推长度受限于 max_len，但参数更少、对 NER 这种相对位置模式有时更稳。
    编码后接 Dropout 作正则。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        t = x.size(1)
        x = x + self.pe[:, :t, :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 主模型：Embedding + TransformerEncoder + 线性发射 + 手写 CRF
# ---------------------------------------------------------------------------

class TransformerCRFNER(nn.Module):
    """
    端到端序列标注模型：Embedding → (+Pos) → TransformerEncoder → Linear → CRF。

    proj 输出 (B,T,C) 的未归一化分数，作为 CRF 一元势；
    LinearChainCRF 学习标签转移与句首/句尾偏置，与一元势相加后形成完整 score(x,y)，
    若外层再包 DistributedDataParallel，对外推理需要使用 unwrap_model(self) 调用 decode_best_tags，
    DDP 包装器默认不转发自定义方法。
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        transition_allow: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_tags = num_tags
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, num_tags)
        self.crf = LinearChainCRF(num_tags, transition_allow=transition_allow)
        self._tag_weights: torch.Tensor | None = None
        self._reset_parameters()

    def set_tag_weights(self, weights: torch.Tensor | None) -> None:
        self._tag_weights = weights

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embed.weight[2:, :])  # 跳过 PAD/UNK 行可选
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        tags: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
            input_ids: (B, T) token id，PAD=0
            lengths: (B,) 每条句真实长度
            tags: 若提供则返回 loss；否则只返回 emissions
        Returns:
            若 tags 非空：标量 loss；否则 emissions (B, T, C)
        """
        # Transformer 约定：True 表示忽略该位置。
        # PAD 的 embedding 虽为 0 向量，仍应用 mask 避免注意力在 padding 上浪费算力并产生无意义依赖。
        pad_mask = input_ids.eq(0)
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        # 线性层给出每个 (batch, 时间, 标签) 的分数，作为 CRF 的一元势
        emissions = self.proj(h)
        if tags is not None:
            return self.crf.batch_neg_log_likelihood(
                emissions, tags, lengths, tag_weights=self._tag_weights
            )
        return emissions

    def decode_best_tags(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[list[int]]:
        """维特比得到每条句最优标签 id 序列。"""
        self.eval()
        with torch.no_grad():
            pad_mask = input_ids.eq(0)
            x = self.embed(input_ids) * math.sqrt(self.d_model)
            x = self.pos(x)
            h = self.encoder(x, src_key_padding_mask=pad_mask)
            emissions = self.proj(h)
            return self.crf.batch_decode(emissions, lengths)


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------

class SentenceNERDataset(Dataset):
    """
    一句 = Dataset 的一个样本，__getitem__ 返回整型 id 序列与长度。
    超长句在 max_len 处截断（与验证阶段一致），避免单句过长拖垮显存。
    """

    def __init__(
        self,
        sentences: list[list[tuple[str, str]]],
        tok_stoi: dict[str, int],
        tag_stoi: dict[str, int],
        max_len: int,
        language: str,
    ) -> None:
        self.samples = sentences
        self.tok_stoi = tok_stoi
        self.tag_stoi = tag_stoi
        self.max_len = max_len
        self.language = language

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], int]:
        sent = self.samples[idx]
        unk = self.tok_stoi["<unk>"]
        ids = [self.tok_stoi.get(normalize_token(w, self.language), unk) for w, _ in sent]
        labs = [self.tag_stoi[t] for _, t in sent]
        L = len(ids)
        if L > self.max_len:
            ids = ids[: self.max_len]
            labs = labs[: self.max_len]
            L = self.max_len
        return ids, labs, L


def collate_fn(batch: list[tuple[list[int], list[int], int]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """
    将一批变长句子对齐到本 batch 内的最大长度 T_max。

    input_ids 中 PAD 位置填 pad_id（与 Embedding(padding_idx=0) 一致，须为 0）。
    tags 的 PAD 段填 0：这些位置不会进入 CRF 的 lengths 计长，故标签值无影响。
    lengths 记录每条句真实长度 L，供 CRF 只在 [:L] 上计算 NLL / 维特比。
    """
    max_t = max(b[2] for b in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_t), pad_id, dtype=torch.long)
    tags = torch.full((batch_size, max_t), 0, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for i, (ids, labs, L) in enumerate(batch):
        lengths[i] = L
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        tags[i, :L] = torch.tensor(labs, dtype=torch.long)
    return {"input_ids": input_ids, "tags": tags, "lengths": lengths}


# ---------------------------------------------------------------------------
# 训练 / 验证集预测
# ---------------------------------------------------------------------------

def _encode_tokens(toks: list[str], tok_stoi: dict[str, int], language: str) -> list[int]:
    unk = tok_stoi["<unk>"]
    return [tok_stoi.get(normalize_token(w, language), unk) for w in toks]


@torch.no_grad()
def decode_sentences(
    model: TransformerCRFNER,
    sentences: list[list[str]],
    tok_stoi: dict[str, int],
    tag_itos: list[str],
    tag_stoi: dict[str, int],
    language: str,
    device: torch.device,
    max_len: int,
    batch_size: int = 16,
) -> list[list[str]]:
    """对 token 列表批量维特比解码，返回标签字符串序列。"""
    was_training = model.training
    model.eval()
    o_tag_idx = tag_stoi.get("O", 0)
    pred_tag_strs: list[list[str]] = []
    try:
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            max_t = min(max((len(s) for s in chunk), default=0), max_len)
            if max_t == 0:
                pred_tag_strs.extend([[] for _ in chunk])
                continue
            input_ids = torch.zeros(len(chunk), max_t, dtype=torch.long)
            lengths = torch.zeros(len(chunk), dtype=torch.long)
            for i, toks in enumerate(chunk):
                ids = _encode_tokens(toks, tok_stoi, language)
                L = min(len(ids), max_len)
                lengths[i] = L
                input_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            id_seqs = model.decode_best_tags(input_ids, lengths)
            for toks, ids_pred, L in zip(chunk, id_seqs, lengths.tolist()):
                L = int(L)
                tags_str = [tag_itos[j] for j in ids_pred[:L]]
                if len(tags_str) < len(toks):
                    tags_str.extend([tag_itos[o_tag_idx]] * (len(toks) - len(tags_str)))
                pred_tag_strs.append(tags_str[: len(toks)])
    finally:
        if was_training:
            model.train()
    return pred_tag_strs


def eval_validation_micro_f1(
    model: TransformerCRFNER,
    val_path: Path,
    val_tokens: list[list[str]],
    gold_sents: list[list[tuple[str, str]]],
    tok_stoi: dict[str, int],
    tag_itos: list[str],
    tag_stoi: dict[str, int],
    language: str,
    device: torch.device,
    max_len: int,
) -> float:
    pred_tags = decode_sentences(
        model, val_tokens, tok_stoi, tag_itos, tag_stoi, language, device, max_len
    )
    y_true: list[str] = []
    y_pred: list[str] = []
    for sent_gold, tags_p in zip(gold_sents, pred_tags):
        for (_, tg), tp in zip(sent_gold, tags_p):
            y_true.append(tg)
            y_pred.append(tp)
    return micro_f1_from_tag_lists(language, y_true, y_pred)


def resolve_lang_hparams(language: str, args: argparse.Namespace) -> dict[str, Any]:
    """按语言给出较稳的训练配置（可通过 --no-lang-tune 关闭）。"""
    if getattr(args, "no_lang_tune", False):
        return {
            "max_len": args.max_len,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }
    if language == "English":
        return {
            "max_len": args.max_len,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            # 与历史最佳验证 F1 对齐：12 epoch、每卡 batch≤64
            "epochs": max(args.epochs, 12),
            "batch_size": min(args.batch_size, 64),
            "lr": 2e-3 if args.lr == 1.5e-3 else args.lr,
        }
    return {
        "max_len": args.max_len,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "epochs": max(args.epochs, 12),
        "batch_size": args.batch_size,
        "lr": 2e-3 if args.lr == 1.5e-3 else args.lr,
    }


def train_one_language(
    language: str,
    ner_root: Path,
    out_dir: Path,
    device: torch.device,
    max_len: int = 256,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 2e-3,
    save_ckpt: Path | None = None,
    *,
    use_ddp: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    use_bio_mask: bool = True,
    use_tag_weights: bool = True,
    eval_each_epoch: bool = True,
    eval_every: int = 1,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    读训练语料 → 建词表/标签表 →（多卡时）按 rank 划分 DataLoader → Transformer+CRF 前向算 NLL → 反传更新 → 仅 rank 0 做验证解码与写盘，避免多进程同时写同一文件。

    参数说明（多卡）：
        batch_size: 每个 GPU / 每个进程上的 batch 大小，不是全局总和。
        use_ddp: 为 True 且 world_size>1 时，使用 DistributedSampler + DDP。
        rank / local_rank / world_size: 来自 setup_distributed()。
    """
    train_path = ner_root / language / "train.txt"
    val_path = ner_root / language / "validation.txt"
    pred_path = out_dir / f"{language}_validation_transformer_crf.txt"

    if rank == 0:
        print(f"[{language}] 读训练: {train_path}")
    train_sents = read_tagged_corpus(train_path)
    val_gold = read_tagged_corpus(val_path)
    tok_stoi, tok_itos, tag_stoi, tag_itos = build_vocab_and_tags(train_sents, language)
    num_tags = len(tag_itos)
    vocab_size = len(tok_itos)

    transition_allow = None
    if use_bio_mask and use_bio_transition_mask(language, tag_itos):
        transition_allow = build_bio_transition_allow(tag_itos)
        if rank == 0:
            print(f"[{language}] 启用 BIO 转移约束")

    ds = SentenceNERDataset(train_sents, tok_stoi, tag_stoi, max_len, language)
    # 多卡：每进程一个 DistributedSampler，保证各 rank 看到的数据子集不重叠且并起来为全量
    sampler: DistributedSampler | None = None
    if use_ddp and world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # 各进程模型结构一致；DDP 会在第一步反向时对齐各卡梯度
    torch.manual_seed(42)
    base = TransformerCRFNER(
        vocab_size=vocab_size,
        num_tags=num_tags,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=4 * d_model,
        dropout=0.15 if language == "English" else 0.1,
        max_len=max(512, max_len + 8),
        transition_allow=transition_allow,
    ).to(device)
    if use_tag_weights:
        tw = compute_tag_weights(train_sents, tag_itos).to(device)
        base.set_tag_weights(tw)

    if use_ddp and world_size > 1:
        model = DDP(
            base,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        model = base

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05
    )
    raw = unwrap_model(model)

    batches_per_epoch = len(dl)
    if rank == 0:
        print(
            f"[{language}] 训练句数={len(ds)}  "
            f"batches/epoch={batches_per_epoch}  "
            f"epochs={epochs}  batch_size={batch_size}  lr={lr}"
        )

    val_tokens = read_tokens_only(val_path) if rank == 0 else []
    best_f1 = -1.0
    best_state: dict[str, Any] | None = None

    model.train()
    for ep in range(1, epochs + 1):
        if sampler is not None:
            # 每个 epoch 使用不同随机划分（否则每 epoch 子集顺序固定）
            sampler.set_epoch(ep)
        total_loss = 0.0
        n_batches = 0
        batch_iter: Any = dl
        if rank == 0:
            batch_iter = tqdm(
                dl,
                desc=f"[{language}] train {ep}/{epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=(ep == epochs),
            )
        for batch in batch_iter:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            tags = batch["tags"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids, lengths, tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            if rank == 0 and isinstance(batch_iter, tqdm):
                batch_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{total_loss / n_batches:.4f}",
                    refresh=False,
                )
        scheduler.step()
        # DDP：各卡只看到自己的 batch 平均 loss；这里仅 rank0 打印本卡均值作参考
        if rank == 0:
            msg = f"[{language}] epoch {ep}/{epochs}  mean_nll={total_loss / max(n_batches, 1):.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            if eval_each_epoch and (ep % eval_every == 0 or ep == epochs):
                f1 = eval_validation_micro_f1(
                    raw,
                    val_path,
                    val_tokens,
                    val_gold,
                    tok_stoi,
                    tag_itos,
                    tag_stoi,
                    language,
                    device,
                    max_len,
                )
                msg += f"  val_micro_f1={f1:.4f}"
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
            print(msg)
        # decode_sentences 会 eval；DDP 下务必恢复 train，否则各卡行为不一致
        model.train()

    if rank == 0 and best_state is not None:
        raw.load_state_dict(best_state)
        print(f"[{language}] 已加载验证 F1 最优权重 (best_micro_f1={best_f1:.4f})")

    # 所有进程先结束训练步，再进入仅 rank0 的解码（其他 rank 在 barrier 等待）
    if use_ddp and world_size > 1:
        dist.barrier()

    meta: dict[str, Any] = {
        "language": language,
        "tok_stoi": tok_stoi,
        "tag_stoi": tag_stoi,
        "tag_itos": tag_itos,
        "max_len": max_len,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "num_tags": num_tags,
        "normalize_lower": language == "English",
        "use_bio_mask": transition_allow is not None,
        "best_val_micro_f1": best_f1,
    }

    if rank == 0:
        print(f"[{language}] 解码验证集 -> {pred_path}")
        pred_tag_strs = decode_sentences(
            raw, val_tokens, tok_stoi, tag_itos, tag_stoi, language, device, max_len
        )
        write_predictions(pred_path, val_tokens, pred_tag_strs)

        if save_ckpt is not None:
            save_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": raw.state_dict(),
                    "meta": meta,
                },
                save_ckpt,
            )
            print(f"[{language}] checkpoint: {save_ckpt}")

    if use_ddp and world_size > 1:
        dist.barrier()

    return pred_path, val_path, meta


def print_check_hint(language: str, gold_path: Path, pred_path: Path) -> None:
    gold_rel = os.path.relpath(gold_path, ner_data_dir())
    pred_rel = os.path.relpath(pred_path, ner_data_dir())
    print(
        "\n--- check.py ---\n"
        f"python -c \"from check import check; check(language='{language}', "
        f"gold_path=r'{gold_rel}', my_path=r'{pred_rel}')\""
    )


def predict_with_checkpoint(
    ckpt_path: Path,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 16,
) -> None:
    """
    单进程推理：加载 train_one_language 保存的 .pt（含 meta 与 state_dict），
    对无标签测试文件逐 batch 维特比解码。

    不参与 DDP；多卡推理可自行按句划分并行多个进程（本函数未封装）。
    """
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)
    meta = ck["meta"]
    language: str = meta.get("language", "English")
    tok_stoi: dict[str, int] = meta["tok_stoi"]
    tag_stoi: dict[str, int] = meta["tag_stoi"]
    tag_itos: list[str] = meta["tag_itos"]
    max_len = int(meta["max_len"])

    transition_allow = None
    if meta.get("use_bio_mask"):
        transition_allow = build_bio_transition_allow(tag_itos)

    model = TransformerCRFNER(
        vocab_size=int(meta["vocab_size"]),
        num_tags=int(meta["num_tags"]),
        d_model=int(meta["d_model"]),
        nhead=int(meta["nhead"]),
        num_layers=int(meta["num_layers"]),
        dim_feedforward=4 * int(meta["d_model"]),
        dropout=0.1,
        max_len=max(512, max_len + 8),
        transition_allow=transition_allow,
    ).to(device)
    model.load_state_dict(ck["model_state"])
    sents = read_tokens_only(input_path)
    pred_tags = decode_sentences(
        model, sents, tok_stoi, tag_itos, tag_stoi, language, device, max_len, batch_size
    )
    write_predictions(output_path, sents, pred_tags)
    print(f"已写出: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="任务三：Transformer + 手写 CRF（支持 torchrun 多卡 DDP）"
    )
    parser.add_argument("--lang", choices=["English", "Chinese", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每块 GPU 上的 batch（DDP 时全局等效约 batch_size×卡数）",
    )
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker 数；Linux 多卡可试 2~4，Windows 建议 0",
    )
    parser.add_argument(
        "--no-lang-tune",
        action="store_true",
        help="关闭按语言自动调整结构/epoch/batch（完全使用命令行参数）",
    )
    parser.add_argument(
        "--no-bio-mask",
        action="store_true",
        help="英文也不使用 BIO 转移硬约束",
    )
    parser.add_argument(
        "--no-tag-weights",
        action="store_true",
        help="关闭实体标签加权 NLL",
    )
    parser.add_argument(
        "--no-epoch-eval",
        action="store_true",
        help="不在每个 epoch 末算验证 F1 / 保存最优权重",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="每 N 个 epoch 做一次验证 F1 并更新最优权重（最后一轮总会验证）",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="若非空，仅 rank0 保存 {English|Chinese}_transformer_crf.pt",
    )
    parser.add_argument("--ckpt-path", type=str, default="", help="仅解码：checkpoint")
    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()

    # ---------- 仅解码：不走分布式 ----------
    if args.ckpt_path or args.input_path or args.output_path:
        if not (args.ckpt_path and args.input_path and args.output_path):
            raise SystemExit("仅解码需同时提供 --ckpt-path --input-path --output-path")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device}")
        predict_with_checkpoint(
            Path(args.ckpt_path),
            Path(args.input_path),
            Path(args.output_path),
            device,
        )
        return

    # ---------- 训练：可选 DDP ----------
    use_ddp, local_rank, rank, world_size, device = setup_distributed()
    if rank == 0:
        print(f"device={device}  use_ddp={use_ddp}  world_size={world_size}")

    ner_root = ner_data_dir()
    out_dir = ner_root / "predictions" / "transformer_crf"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp and world_size > 1:
        dist.barrier()

    save_root = Path(args.save_dir) if args.save_dir else None

    try:
        langs = ["English", "Chinese"] if args.lang == "both" else [args.lang]
        for lang in langs:
            ck = None
            if save_root is not None and rank == 0:
                save_root.mkdir(parents=True, exist_ok=True)
                ck = save_root / f"{lang}_transformer_crf.pt"
            hp = resolve_lang_hparams(lang, args)
            if rank == 0:
                print(f"[{lang}] hparams: {hp}")
            pred_path, gold_path, _ = train_one_language(
                language=lang,
                ner_root=ner_root,
                out_dir=out_dir,
                device=device,
                max_len=hp["max_len"],
                d_model=hp["d_model"],
                nhead=hp["nhead"],
                num_layers=hp["num_layers"],
                epochs=hp["epochs"],
                batch_size=hp["batch_size"],
                lr=hp["lr"],
                save_ckpt=ck,
                use_ddp=use_ddp,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                num_workers=args.num_workers,
                use_bio_mask=not args.no_bio_mask,
                use_tag_weights=not args.no_tag_weights,
                eval_each_epoch=not args.no_epoch_eval,
                eval_every=max(1, args.eval_every),
            )
            if rank == 0:
                print_check_hint(lang, gold_path, pred_path)
    finally:
        cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
