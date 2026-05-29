"""
BERT + 手写线性链 CRF 的 NER

与 transformer_crf_ner.py 的区别：
- 编码器换为 HuggingFace 预训练 BERT（英文 bert-base-cased，中文 bert-base-chinese）
- CRF 仍使用同目录 linear_chain_crf.py（手写、可端到端反传）
- 标签对齐：每个原词/字取 BERT 子词序列的首子词 hidden，再在词级序列上跑 CRF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from linear_chain_crf import LinearChainCRF

# 复用 transformer_crf_ner 的数据与评测工具（避免重复实现）
from transformer_crf_ner import (
    build_bio_transition_allow,
    build_vocab_and_tags,
    compute_tag_weights,
    micro_f1_from_tag_lists,
    ner_data_dir,
    normalize_token,
    print_check_hint,
    read_tagged_corpus,
    read_tokens_only,
    use_bio_transition_mask,
    write_predictions,
)

DEFAULT_BERT = {
    "English": "bert-base-cased",
    "Chinese": "bert-base-chinese",
}


def default_bert_name(language: str) -> str:
    if language not in DEFAULT_BERT:
        raise ValueError("language 须为 English 或 Chinese")
    return DEFAULT_BERT[language]


def first_subword_indices(word_ids: list[int | None], num_words: int) -> list[int]:
    """从 tokenizer.word_ids() 得到每个原词位置在 BERT 序列中的首子词下标。"""
    indices: list[int] = []
    prev: int | None = None
    for t, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev:
            indices.append(t)
            prev = wid
        if len(indices) >= num_words:
            break
    return indices


class BertCRFNER(nn.Module):
    """BERT 编码 → 词级 hidden → Linear 发射 → 手写 CRF。"""

    def __init__(
        self,
        bert_name: str,
        num_tags: int,
        dropout: float = 0.1,
        transition_allow: torch.Tensor | None = None,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        self.bert_name = bert_name
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_tags)
        self.crf = LinearChainCRF(num_tags, transition_allow=transition_allow)
        self._tag_weights: torch.Tensor | None = None

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def set_tag_weights(self, weights: torch.Tensor | None) -> None:
        self._tag_weights = weights

    def _word_level_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids_batch: list[list[int | None]],
        lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        """返回长度为 batch 的列表，每项 (L_i, C)。"""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)
        emissions_list: list[torch.Tensor] = []
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            L = int(lengths[b].item())
            if L <= 0:
                emissions_list.append(hidden.new_zeros(0, self.crf.num_tags))
                continue
            idx = first_subword_indices(word_ids_batch[b], L)
            if len(idx) < L:
                L = len(idx)
            if L == 0:
                emissions_list.append(hidden.new_zeros(0, self.crf.num_tags))
                continue
            h = self.drop(hidden[b, idx[:L]])
            emissions_list.append(self.classifier(h))
        return emissions_list

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids_batch: list[list[int | None]],
        lengths: torch.Tensor,
        tags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emissions_list = self._word_level_emissions(
            input_ids, attention_mask, word_ids_batch, lengths
        )
        if tags is None:
            raise ValueError("训练需提供 tags")

        total = input_ids.new_zeros(())
        count = 0
        for b, emit in enumerate(emissions_list):
            L = emit.size(0)
            if L <= 0:
                continue
            total = total + self.crf.neg_log_likelihood(
                emit, tags[b, :L], tag_weights=self._tag_weights
            )
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def decode_best_tags(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids_batch: list[list[int | None]],
        lengths: torch.Tensor,
    ) -> list[list[int]]:
        self.eval()
        emissions_list = self._word_level_emissions(
            input_ids, attention_mask, word_ids_batch, lengths
        )
        out: list[list[int]] = []
        for emit in emissions_list:
            if emit.size(0) == 0:
                out.append([])
            else:
                out.append(self.crf.viterbi_decode(emit))
        return out


class BertNERDataset(Dataset):
    def __init__(
        self,
        sentences: list[list[tuple[str, str]]],
        tag_stoi: dict[str, int],
        max_words: int,
        language: str,
    ) -> None:
        self.samples = sentences
        self.tag_stoi = tag_stoi
        self.max_words = max_words
        self.language = language

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int], int]:
        sent = self.samples[idx]
        words = [normalize_token(w, self.language) for w, _ in sent]
        labs = [self.tag_stoi[t] for _, t in sent]
        L = len(words)
        if L > self.max_words:
            words = words[: self.max_words]
            labs = labs[: self.max_words]
            L = self.max_words
        return words, labs, L


def make_collate_fn(tokenizer, max_bert_len: int):
    def collate(batch: list[tuple[list[str], list[int], int]]) -> dict[str, Any]:
        words_batch = [b[0] for b in batch]
        tags_batch = [b[1] for b in batch]
        lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)

        enc = tokenizer(
            words_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_bert_len,
            return_tensors="pt",
        )
        max_w = int(lengths.max().item())
        tags = torch.zeros(len(batch), max_w, dtype=torch.long)
        for i, (lab, L) in enumerate(zip(tags_batch, lengths.tolist())):
            tags[i, :L] = torch.tensor(lab[:L], dtype=torch.long)

        word_ids_batch = [enc.word_ids(i) for i in range(len(batch))]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "tags": tags,
            "lengths": lengths,
            "word_ids_batch": word_ids_batch,
        }

    return collate


@torch.no_grad()
def decode_sentences_bert(
    model: BertCRFNER,
    sentences: list[list[str]],
    tag_itos: list[str],
    tag_stoi: dict[str, int],
    language: str,
    device: torch.device,
    max_words: int,
    batch_size: int = 8,
) -> list[list[str]]:
    was_training = model.training
    model.eval()
    o_idx = tag_stoi.get("O", 0)
    pred: list[list[str]] = []
    try:
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            words_batch = [[normalize_token(w, language) for w in s] for s in chunk]
            lengths_list = [min(len(w), max_words) for w in words_batch]
            for i, w in enumerate(words_batch):
                if len(w) > max_words:
                    words_batch[i] = w[:max_words]
            lengths = torch.tensor(lengths_list, dtype=torch.long)

            enc = model.tokenizer(
                words_batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=max_words + 32,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            word_ids_batch = [enc.word_ids(i) for i in range(len(chunk))]

            id_seqs = model.decode_best_tags(
                input_ids, attention_mask, word_ids_batch, lengths.to(device)
            )
            for toks, ids_pred, L in zip(chunk, id_seqs, lengths_list):
                tags_str = [tag_itos[j] for j in ids_pred[:L]]
                if len(tags_str) < len(toks):
                    tags_str.extend([tag_itos[o_idx]] * (len(toks) - len(tags_str)))
                pred.append(tags_str[: len(toks)])
    finally:
        if was_training:
            model.train()
    return pred


def train_one_language(
    language: str,
    ner_root: Path,
    out_dir: Path,
    device: torch.device,
    bert_name: str,
    max_words: int = 128,
    max_bert_len: int = 256,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    freeze_bert: bool = False,
    save_ckpt: Path | None = None,
    use_bio_mask: bool = True,
    use_tag_weights: bool = True,
) -> tuple[Path, Path]:
    train_path = ner_root / language / "train.txt"
    val_path = ner_root / language / "validation.txt"
    pred_path = out_dir / f"{language}_validation_bert_crf.txt"

    print(f"[{language}] BERT={bert_name}")
    train_sents = read_tagged_corpus(train_path)
    val_gold = read_tagged_corpus(val_path)
    _, _, tag_stoi, tag_itos = build_vocab_and_tags(train_sents, language)
    num_tags = len(tag_itos)

    transition_allow = None
    if use_bio_mask and use_bio_transition_mask(language, tag_itos):
        transition_allow = build_bio_transition_allow(tag_itos)
        print(f"[{language}] BIO 转移约束已启用")

    model = BertCRFNER(
        bert_name=bert_name,
        num_tags=num_tags,
        dropout=0.1,
        transition_allow=transition_allow,
        freeze_bert=freeze_bert,
    ).to(device)
    if use_tag_weights:
        model.set_tag_weights(compute_tag_weights(train_sents, tag_itos).to(device))

    ds = BertNERDataset(train_sents, tag_stoi, max_words, language)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(model.tokenizer, max_bert_len),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    bert_params = list(model.bert.parameters())
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("bert.")
    ]
    opt = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": lr * (0.2 if freeze_bert else 1.0)},
            {"params": head_params, "lr": lr * 5.0},
        ],
        weight_decay=0.01,
    )

    val_tokens = read_tokens_only(val_path)
    best_f1 = -1.0
    best_state: dict[str, Any] | None = None

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(dl, desc=f"[{language}] bert train {ep}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tags = batch["tags"].to(device)
            lengths = batch["lengths"].to(device)
            word_ids_batch = batch["word_ids_batch"]

            opt.zero_grad(set_to_none=True)
            loss = model(input_ids, attention_mask, word_ids_batch, lengths, tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / n_batches:.4f}")

        f1 = _eval_bert_f1(
            model, val_gold, val_tokens, tag_itos, tag_stoi, language, device, max_words
        )
        print(
            f"[{language}] epoch {ep}/{epochs}  mean_nll={total_loss / max(n_batches, 1):.4f}  "
            f"val_micro_f1={f1:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{language}] 已加载最优验证权重 best_micro_f1={best_f1:.4f}")

    pred_tags = decode_sentences_bert(
        model, val_tokens, tag_itos, tag_stoi, language, device, max_words
    )
    write_predictions(pred_path, val_tokens, pred_tags)

    meta = {
        "language": language,
        "bert_name": bert_name,
        "tag_stoi": tag_stoi,
        "tag_itos": tag_itos,
        "max_words": max_words,
        "max_bert_len": max_bert_len,
        "num_tags": num_tags,
        "use_bio_mask": transition_allow is not None,
        "best_val_micro_f1": best_f1,
        "model_type": "bert_crf",
    }
    if save_ckpt is not None:
        save_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "meta": meta}, save_ckpt)
        print(f"[{language}] checkpoint: {save_ckpt}")

    return pred_path, val_path


def _eval_bert_f1(
    model: BertCRFNER,
    gold_sents: list[list[tuple[str, str]]],
    val_tokens: list[list[str]],
    tag_itos: list[str],
    tag_stoi: dict[str, int],
    language: str,
    device: torch.device,
    max_words: int,
) -> float:
    pred = decode_sentences_bert(
        model, val_tokens, tag_itos, tag_stoi, language, device, max_words
    )
    y_true, y_pred = [], []
    for sent, tags_p in zip(gold_sents, pred):
        for (_, tg), tp in zip(sent, tags_p):
            y_true.append(tg)
            y_pred.append(tp)
    return micro_f1_from_tag_lists(language, y_true, y_pred)


def predict_with_checkpoint(
    ckpt_path: Path,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 8,
) -> None:
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)
    meta = ck["meta"]
    language = meta["language"]
    tag_stoi = meta["tag_stoi"]
    tag_itos = meta["tag_itos"]
    max_words = int(meta["max_words"])
    bert_name = meta.get("bert_name") or default_bert_name(language)

    transition_allow = None
    if meta.get("use_bio_mask"):
        transition_allow = build_bio_transition_allow(tag_itos)

    model = BertCRFNER(
        bert_name=bert_name,
        num_tags=len(tag_itos),
        transition_allow=transition_allow,
    ).to(device)
    model.load_state_dict(ck["model_state"])

    sents = read_tokens_only(input_path)
    pred = decode_sentences_bert(
        model, sents, tag_itos, tag_stoi, language, device, max_words, batch_size
    )
    write_predictions(output_path, sents, pred)
    print(f"已写出: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT + 手写 CRF（NER）")
    parser.add_argument("--lang", choices=["English", "Chinese", "both"], default="both")
    parser.add_argument("--bert-model", type=str, default="", help="覆盖默认 HF 模型名")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-words", type=int, default=128, help="原词/字序列最大长度")
    parser.add_argument("--max-bert-len", type=int, default=256, help="BERT 输入最大 token 数")
    parser.add_argument("--freeze-bert", action="store_true", help="冻结 BERT，只训 CRF+分类头")
    parser.add_argument("--no-bio-mask", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="task3_transformer_crf/checkpoints_bert",
    )
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()

    if args.ckpt_path or args.input_path or args.output_path:
        if not (args.ckpt_path and args.input_path and args.output_path):
            raise SystemExit("仅解码需 --ckpt-path --input-path --output-path")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predict_with_checkpoint(
            Path(args.ckpt_path), Path(args.input_path), Path(args.output_path), device
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    ner_root = ner_data_dir()
    out_dir = ner_root / "predictions" / "bert_crf"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_root = Path(args.save_dir) if args.save_dir else None

    langs = ["English", "Chinese"] if args.lang == "both" else [args.lang]
    for lang in langs:
        bert_name = args.bert_model or default_bert_name(lang)
        ck = None
        if save_root is not None:
            save_root.mkdir(parents=True, exist_ok=True)
            ck = save_root / f"{lang}_bert_crf.pt"
        pred_path, gold_path = train_one_language(
            language=lang,
            ner_root=ner_root,
            out_dir=out_dir,
            device=device,
            bert_name=bert_name,
            max_words=args.max_words,
            max_bert_len=args.max_bert_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_bert=args.freeze_bert,
            save_ckpt=ck,
            use_bio_mask=not args.no_bio_mask,
        )
        print_check_hint(lang, gold_path, pred_path)


if __name__ == "__main__":
    main()
