"""
按 CRFsuite 风格模板（如 template_for_crf.utf8）从观测表生成每个位置的字符串特征。

中文 NER 一字一行时，列 0 即为当前字；边界 %x 越界用 __BOS__ / __EOS__ 占位，
与课堂分词 CRF 特征表约定一致。sklearn-crfsuite 将 (键, 值) 展开为稀疏二值特征。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TemplateLine:
    """一行模板，如 U06:%x[-1,0]/%x[0,0]。"""

    family: str  # "U" or "B"
    tid: str  # e.g. "U06"
    segments: tuple[tuple[int, int], ...]  # (相对行偏移, 列) 序列


_pat = re.compile(r"%x\[(-?\d+),(\d+)\]")


def parse_template(path: str | Path) -> list[TemplateLine]:
    """解析 CRFsuite 模板文件，忽略空行与 # 注释。"""
    path = Path(path)
    out: list[TemplateLine] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            tid_part, rest = line.split(":", 1)
            tid_part = tid_part.strip()
            rest = rest.strip()
            if not tid_part or tid_part[0] not in "UB":
                continue
            family = tid_part[0]
            chunks = [c.strip() for c in rest.split("/") if c.strip()]
            segs: list[tuple[int, int]] = []
            for ch in chunks:
                m = _pat.fullmatch(ch)
                if not m:
                    raise ValueError(f"无法解析模板片段: {ch!r} (行: {line})")
                segs.append((int(m.group(1)), int(m.group(2))))
            out.append(TemplateLine(family=family, tid=tid_part, segments=tuple(segs)))
    return out


def _obs_char(sentence: list[str], i: int, row_off: int, col: int) -> str:
    """取相对当前位置 i 的观测格 (i+row_off, col)；中文单列字。"""
    if col != 0:
        raise ValueError("本作业模板仅使用第 0 列（字）")
    j = i + row_off
    if j < 0:
        return "__BOS__"
    if j >= len(sentence):
        return "__EOS__"
    return sentence[j]


def apply_templates(
    sentence: list[str],
    i: int,
    templates: list[TemplateLine],
) -> dict[str, str]:
    """
    对句子 sentence 的第 i 个位置，按模板列表生成特征字典。
    键为模板 id（U00/B03…），值为实例化后的观测串（可能含 '/' 连接多格）。
    """
    feats: dict[str, str] = {}
    for tl in templates:
        parts = [_obs_char(sentence, i, r, c) for r, c in tl.segments]
        feats[tl.tid] = "/".join(parts)
    feats["bias"] = "1.0"
    return feats


def sentence_to_features(
    sentence: list[str],
    templates: list[TemplateLine],
) -> list[dict[str, str]]:
    return [apply_templates(sentence, pos, templates) for pos in range(len(sentence))]
