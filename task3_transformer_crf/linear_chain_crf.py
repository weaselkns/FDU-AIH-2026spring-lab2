"""
手写线性链 CRF（Conditional Random Field）层
================================================

本模块不调用任何第三方 CRF 库，仅用 torch 实现一阶链式 CRF 的核心算法，
并与上游神经网络输出的发射分数（emissions）联合训练。

---------------------------------------------------------------------------
1. 线性链 CRF 在序列标注里的作用
---------------------------------------------------------------------------
给定输入序列 x = (x_1,...,x_L)（此处 x 已由 Transformer 编码，
发射分数 emissions[t, k] 表示在位置 t 打标签 k 的「一元势」），
定义标注序列 y = (y_1,...,y_L) 的非规范化得分（在 log 域更易数值稳定）：

    score(x, y) = start[y_1] + Σ_t emit_t(y_t) + Σ_t trans(y_{t-1}, y_t) + end[y_L]

其中 start[*]、end[*] 处理句首/句尾与标签图的衔接；trans 为一阶转移势。归一化条件分布为：

    P(y|x) = exp(score(x,y)) / Z(x),   Z(x) = Σ_y' exp(score(x,y'))

训练目标常为负对数似然：

    NLL = -log P(y_gold|x) = log Z(x) - score(x, y_gold)

log Z(x) 用前向算法（log-sum-exp 形式）；score 对金标路径直接求和即可反传。
解码用维特比：把 logsumexp 换成 max 并记录回溯指针。

---------------------------------------------------------------------------
2. 与张量形状
---------------------------------------------------------------------------
emissions: (L, C) 单句，C = num_tags。
batch_neg_log_likelihood：对 batch 维做 for 循环，逐句调用单句 CRF，与变长序列天然兼容，避免复杂 padding mask 在 CRF 内部的边界错误。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """
    一阶线性链 CRF 层。

    对外接口：
    neg_log_likelihood / batch_neg_log_likelihood：供训练计算 NLL；
    viterbi_decode / batch_decode：供推理得到最优标签序列。
    """

    def __init__(
        self,
        num_tags: int,
        transition_allow: torch.Tensor | None = None,
    ) -> None:
        """
        参数：
            num_tags: 标签种类数 C（如 BIO/BMESO 展开后的标签表大小）。
            transition_allow: 可选 (C, C) bool，True 表示允许的标签转移（用于 BIO 等硬约束）。
        """
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j]: 上一时刻标签 i → 当前时刻标签 j 的对数势（未约束实数）
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.02)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.02)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.02)
        if transition_allow is not None:
            if transition_allow.shape != (num_tags, num_tags):
                raise ValueError("transition_allow 须为 (num_tags, num_tags)")
            self.register_buffer("transition_allow", transition_allow.bool())
        else:
            self.transition_allow = None

    def _masked_transitions(self) -> torch.Tensor:
        trans = self.transitions
        if self.transition_allow is not None:
            trans = trans.masked_fill(~self.transition_allow, -1e4)
        return trans

    def _forward_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        """
        计算配分函数 log Z(x) = log Σ_y exp(score(x,y))。

        前向递推（log 域，避免下溢）：

            α[0, j] = start[j] + emit[0, j]
            α[t, j] = log Σ_i exp( α[t-1, i] + trans[i, j] + emit[t, j] )
            log Z   = log Σ_j exp( α[L-1, j] + end[j] )

        在 PyTorch 中用 torch.logsumexp 实现 log Σ exp。

        参数：
            emissions: 形状 (L, C)，L 为句长，C 为标签数。
        Returns:
            标量 log Z，对 emissions 与 CRF 参数可微（便于反传）。
        """
        if emissions.dim() != 2:
            raise ValueError("emissions 应为 (L, C)")
        seq_len, num_tags = emissions.shape
        if num_tags != self.num_tags:
            raise ValueError("最后一维须等于 num_tags")

        # α 向量长度 C：表示「当前时刻落在各标签」上的累积 log 权重
        alpha = self.start_transitions + emissions[0]

        if seq_len == 1:
            return torch.logsumexp(alpha + self.end_transitions, dim=0)

        for t in range(1, seq_len):
            # prev[i, j] = α[t-1, i] + trans[i, j]，对 i 做 logsumexp 得到 α[t, j] 前的部分
            emit_t = emissions[t].unsqueeze(0)  # (1, C)，广播加到各 j
            prev = alpha.unsqueeze(1) + self._masked_transitions()  # (C, C)
            alpha = torch.logsumexp(prev, dim=0) + emit_t.squeeze(0)

        return torch.logsumexp(alpha + self.end_transitions, dim=0)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """
        金标路径的非规范化对数得分 score(x, y)（不是 log 概率，未减 log Z）。

        按时间步展开：

            score = start[y0] + emit[0,y0]
                  + Σ_{t=1}^{L-1} ( trans[y_{t-1}, y_t] + emit[t, y_t] )
                  + end[y_{L-1}]

        参数：
            emissions: (L, C)
            tags: (L,) 整型，取值 0 .. C-1
        Returns:
            标量得分。
        """
        seq_len, num_tags = emissions.shape
        if tags.numel() != seq_len:
            raise ValueError("tags 长度必须与 emissions 一致")

        trans = self._masked_transitions()
        score = self.start_transitions[tags[0]] + emissions[0, tags[0]]
        for t in range(1, seq_len):
            score = score + trans[tags[t - 1], tags[t]] + emissions[t, tags[t]]
        score = score + self.end_transitions[tags[-1]]
        return score

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        tag_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        单句负对数条件似然：

            NLL = -log P(y|x) = log Z(x) - score(x, y)

        对 emissions（由上游网络产生）与 CRF 自身参数均可反传。
        tag_weights: 可选 (C,) 对金标路径上各位置标签权重求平均后缩放 NLL。
        """
        log_z = self._forward_partition(emissions)
        gold = self._score_sentence(emissions, tags)
        nll = log_z - gold
        if tag_weights is not None:
            nll = nll * tag_weights[tags].mean().clamp(min=0.1)
        return nll

    def viterbi_decode(self, emissions: torch.Tensor) -> list[int]:
        """
        维特比解码：y* = argmax_y score(x, y)（同样不含 Z，因 Z 对 y 为常数）。

        递推与 _forward_partition 类似，只是把 logsumexp 换成 max，
        并记录 back[t, j] 回溯指针以便还原整条路径。
        """
        seq_len, num_tags = emissions.shape
        score = torch.full(
            (seq_len, num_tags),
            -1e30,
            dtype=emissions.dtype,
            device=emissions.device,
        )
        back = torch.zeros((seq_len, num_tags), dtype=torch.long, device=emissions.device)

        score[0] = self.start_transitions + emissions[0]
        for t in range(1, seq_len):
            prev = score[t - 1].unsqueeze(1) + self._masked_transitions()
            best_val, best_idx = prev.max(dim=0)
            score[t] = best_val + emissions[t]
            back[t] = best_idx

        best_last = torch.argmax(score[-1] + self.end_transitions)
        tags_rev: list[int] = [int(best_last.item())]
        idx = int(best_last.item())
        for t in range(seq_len - 1, 0, -1):
            idx = int(back[t, idx].item())
            tags_rev.append(idx)
        tags_rev.reverse()
        return tags_rev

    def batch_neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        lengths: torch.Tensor,
        tag_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        对一个 mini-batch 求平均 NLL。

        每条样本只取 [:L] 有效长度（L = lengths[b]），padding 段不参与
        CRF；这与 Transformer 侧 ``src_key_padding_mask`` 的语义一致。

        参数：
            emissions: (B, T, C)
            tags: (B, T)，padding 位置可为任意值
            lengths: (B,)，每条句真实长度
        Returns:
            标量 mean_b NLL_b
        """
        batch_size = emissions.size(0)
        total = emissions.new_zeros(())
        count = 0
        for b in range(batch_size):
            L = int(lengths[b].item())
            if L <= 0:
                continue
            total = total + self.neg_log_likelihood(
                emissions[b, :L], tags[b, :L], tag_weights=tag_weights
            )
            count += 1
        if count == 0:
            return total
        return total / count

    def batch_decode(
        self,
        emissions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[list[int]]:
        """
        对 batch 中每条序列分别维特比解码，返回标签下标列表的列表。
        """
        out: list[list[int]] = []
        batch_size = emissions.size(0)
        for b in range(batch_size):
            L = int(lengths[b].item())
            if L <= 0:
                out.append([])
                continue
            out.append(self.viterbi_decode(emissions[b, :L]))
        return out
