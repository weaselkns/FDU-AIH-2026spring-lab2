"""
任务二：基于线性链条件随机场（CRF）的命名实体识别

一、与 HMM 的对比
HMM 生成式：假设观测 x_t 在给定 y_t 下条件独立，发射用 P(x|y)，转移用 P(y_t|y_{t-1})，解码用维特比。
线性链 CRF 判别式：直接建模 P(y|x)，对整条标注序列定义 score(x,y) = Σ_t 一元项 f(x,t,y_t) + Σ_t 二元项 g(y_{t-1},y_t)，
其中一元项在本实现里由手工模板特征经线性权重加权得到（sklearn-crfsuite 内部用结构化感知机 / L-BFGS 等优化）。
不要求 x_t 条件独立，因此可把窗口、词形、邻域词等任意特征塞进 x 侧。

二、本文件用到的库
sklearn-crfsuite 是对 CRFsuite 的 Python 封装：
训练算法为 L-BFGS，带 L1(c1)、L2(c2) 正则；解码为维特比（与 HMM 同为动态规划，但势函数来自特征权重而非生成概率）。
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import unicodedata
from pathlib import Path

# sklearn-crfsuite：对线性链 CRF 的高效训练/维特比解码封装
from sklearn_crfsuite import CRF


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """仓库根目录（本文件位于 pj2/part2/）。"""
    return Path(__file__).resolve().parent.parent.parent


def ner_data_dir() -> Path:
    """课程 NER 数据根目录。"""
    return project_root() / "NER"


# ---------------------------------------------------------------------------
# 数据读写（CoNLL：token tag，句间空行）
# ---------------------------------------------------------------------------

def read_tagged_corpus(path: str | Path) -> list[list[tuple[str, str]]]:
    """
    读取带 gold 标签的语料文件。

    约定与课程数据一致：非空行 ``词 标签``（空白分隔）；连续空行视为多分句；
    句末空行可有可无，本函数在文件末尾若仍有累积句也会输出。

    参数：
        path: train.txt / validation.txt 等 UTF-8 文本路径。
    Returns:
        sentences[k] 为第 k 句的 (token, tag) 列表。
    """
    path = Path(path)
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
                continue
            cur.append((parts[0], parts[1]))
    if cur:
        sentences.append(cur)
    return sentences


def read_tokens_only(path: str | Path) -> list[list[str]]:
    """读取待预测句子，每行取第一列为 token。"""
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
    """写出 token + 预测标签，句末空行。"""
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
# 英文特征：词级 + 形状 + 邻域（CRFsuite）
# ---------------------------------------------------------------------------

def _word_shape(word: str, max_len: int = 8) -> str:
    """
    将词压缩为「形状」串：大写 X、小写 x、数字 d、其它记为原字符类别 c。
    用于捕获如 NYSE、McDonald 等模式，且不随具体词形爆炸词表。
    """
    out: list[str] = []
    for ch in word:
        if ch.isdigit():
            out.append("d")
        elif ch.islower():
            out.append("x")
        elif ch.isupper():
            out.append("X")
        else:
            out.append("c")
    s = "".join(out)
    return s[:max_len] if len(s) > max_len else s


def token_features_english(sentence: list[str], i: int) -> dict[str, str]:
    """
    为英文句中第 i 个词构造一组局部特征（全部用字符串，满足 CRFsuite 输入约定）。
    当前词是什么、邻居是什么、是否像数字/公司名缩写/连字符词等，这些信号对 PER/ORG/LOC/MISC 的区分往往比“仅当前词”强。
    """
    w = sentence[i]
    lower = w.lower()
    prev1 = sentence[i - 1] if i > 0 else "__BOS__"
    next1 = sentence[i + 1] if i + 1 < len(sentence) else "__EOS__"
    prev2 = sentence[i - 2] if i > 1 else "__BOS2__"
    next2 = sentence[i + 2] if i + 2 < len(sentence) else "__EOS2__"

    feats: dict[str, str] = {
        "bias": "1.0",  # 全局偏置（截距），每个位置恒定激活
        "w.lower": lower,  # 当前词小写全形，强记忆、OOV 弱
        "w[-3:]": lower[-3:] if len(lower) >= 3 else lower,  # 词尾最多 3 字符（后缀模式）
        "w[-2:]": lower[-2:] if len(lower) >= 2 else lower,  # 词尾 2 字符
        "w[:3]": lower[:3],  # 词头 3 字符（前缀模式）
        "w[:2]": lower[:2],  # 词头 2 字符
        "w.shape": _word_shape(w),  # 字形串 X/x/d/c，泛化大小写专名等
        "prev1": prev1.lower(),  # 左邻 1 词（小写）
        "next1": next1.lower(),  # 右邻 1 词（小写）
        "prev2": prev2.lower(),  # 左邻 2 词（小写）
        "next2": next2.lower(),  # 右邻 2 词（小写）
        "pos.ratio": f"{i / max(len(sentence) - 1, 1):.3f}",  # 在句中相对位置 0~1
    }

    # 布尔型也编码为离散取值的字符串，便于线性权重复用
    feats["BOS"] = str(i == 0)  # 是否句首词
    feats["EOS"] = str(i == len(sentence) - 1)  # 是否句末词
    feats["w.isdigit"] = str(w.isdigit())  # 是否整词为数字
    feats["w.isalpha"] = str(w.isalpha())  # 是否全为字母
    feats["w.isupper"] = str(w.isupper() and w.isalpha())  # 是否全大写词（如 NYSE）
    feats["w.istitle"] = str(w.istitle())  # 是否首字母大写（Title Case）
    feats["hyphen"] = str("-" in w)  # 是否含连字符
    feats["dot"] = str("." in w)  # 是否含句点（缩写 U.S. 等）

    # 常见数字/序数模式（英文数据里较常见）
    feats["re_roman"] = str(bool(re.fullmatch(r"[IVXLC]+", w.upper())))  # 是否罗马数字样式
    feats["re_digits"] = str(bool(re.search(r"\d", w)))  # 是否含任意数字字符

    return feats


def sentence_to_features_english(sentence: list[str]) -> list[dict[str, str]]:
    """
    整条英文句：每个位置一个特征字典。
    例子：
        输入: ["Rangarajan", "said"]
        输出: [
                { "w.lower": "rangarajan", "BOS": "True", ... },   # i=0
                { "w.lower": "said", "BOS": "False", "prev1": "rangarajan", ... },  # i=1
              ]
    """
    return [token_features_english(sentence, i) for i in range(len(sentence))]


def sentence_to_labels(sentence: list[tuple[str, str]]) -> list[str]:
    """
    从带标签句子抽出标签序列。
    例子：
        sentence = [("Rangarajan", "B-PER"), ("said", "O")]
        sentence_to_labels(sentence) → ["B-PER", "O"]
    """
    return [tag for _, tag in sentence]


def sentence_to_tokens(sentence: list[tuple[str, str]]) -> list[str]:
    """
    从带标签句子抽出词序列。
    例子：
        sentence = [("Rangarajan", "B-PER"), ("said", "O")]
        sentence_to_tokens(sentence) → ["Rangarajan", "said"]
    """
    return [w for w, _ in sentence]


# ---------------------------------------------------------------------------
# 中文特征：字级窗口 + 类型（BMESO 标签体系）
# ---------------------------------------------------------------------------

def _char_category(ch: str) -> str:
    """粗粒度 Unicode 类别，帮助区分标点、字母、汉字等。"""
    if ch.isspace():
        return "Z"
    cat = unicodedata.category(ch)[0]
    return cat


def token_features_chinese(sentence: list[str], i: int) -> dict[str, str]:
    """
    中文 NER 数据一般为一字一行，以字为 token，用左右邻字构造特征。
    双字、三字片段对识别姓名、机构尾缀（如“公司”“银行”）特别有效。
    """
    # 当前字与相邻字
    c = sentence[i]
    p1 = sentence[i - 1] if i > 0 else "__BOS__"
    n1 = sentence[i + 1] if i + 1 < len(sentence) else "__EOS__"
    p2 = sentence[i - 2] if i > 1 else "__BOS2__"
    n2 = sentence[i + 2] if i + 2 < len(sentence) else "__EOS2__"
    
    feats: dict[str, str] = {
        "bias": "1.0",  # 全局偏置（截距），每个位置恒定激活
        "c": c,  # 当前字
        "p1": p1,  # 左邻 1 字（句首为 __BOS__）
        "n1": n1,  # 右邻 1 字（句末为 __EOS__）
        "p2": p2,  # 左邻 2 字（不足为 __BOS2__）
        "n2": n2,  # 右邻 2 字（不足为 __EOS2__）
        "p1+c": f"{p1}+{c}",  # 左 1 字 + 当前字（二字组）
        "c+n1": f"{c}+{n1}",  # 当前字 + 右 1 字（二字组）
        "p2+p1": f"{p2}+{p1}",  # 左侧二字（不含当前字）
        "n1+n2": f"{n1}+{n2}",  # 右侧二字（不含当前字，如「公司」尾缀）
        "p2+p1+c": f"{p2}+{p1}+{c}",  # 左二 + 当前（三字窗，实体左端）
        "c+n1+n2": f"{c}+{n1}+{n2}",  # 当前 + 右二（三字窗，实体右端/尾缀）
        "cat": _char_category(c),  # 当前字 Unicode 粗类别（L/N/P/Z 等）
    }
    feats["c.isdigit"] = str(c.isdigit())  # 当前字是否为数字
    feats["c.isascii"] = str(c.isascii())  # 当前字是否为 ASCII（英文/半角符号）
    feats["BOS"] = str(i == 0)  # 是否句首字
    feats["EOS"] = str(i == len(sentence) - 1)  # 是否句末字
    feats["pos.ratio"] = f"{i / max(len(sentence) - 1, 1):.3f}"  # 在句中相对位置 0~1
    return feats


def sentence_to_features_chinese(sentence: list[str]) -> list[dict[str, str]]:
    """整条中文句：每个位置一个特征字典。"""
    return [token_features_chinese(sentence, i) for i in range(len(sentence))]


# ---------------------------------------------------------------------------
# CRF 训练：sklearn_crfsuite 封装 L-BFGS + 转移特征 all_transitions
# ---------------------------------------------------------------------------

def build_xy(
    sentences: list[list[tuple[str, str]]],
    language: str,
) -> tuple[list[list[dict[str, str]]], list[list[str]]]:
    """
    将原始标注语料转换为 sklearn_crfsuite.CRF.fit 需要的 (X, y)。

    X：长度为句数的列表；X[i][t] 为第 i 句第 t 个 token 的特征字典
      （键值均为字符串，值为离散化后的字符串，便于稀疏二值特征展开）。
    y：与 X 对齐的标签字符串列表（CRF 不在内部维护 tag→id，转移特征在标签字符串空间上学习）。
    language：仅用于选择英文或中文的特征模板函数。
    """
    X: list[list[dict[str, str]]] = []
    y: list[list[str]] = []
    if language == "English":
        fe = sentence_to_features_english
    elif language == "Chinese":
        fe = sentence_to_features_chinese
    else:
        raise ValueError("language 必须是 English 或 Chinese")

    for sent in sentences:
        toks = sentence_to_tokens(sent)
        labs = sentence_to_labels(sent)
        if not toks:
            continue
        X.append(fe(toks))
        y.append(labs)
    return X, y


def make_crf(
    c1: float = 0.08,
    c2: float = 0.4,
    max_iterations: int = 80,
    all_possible_transitions: bool = True,
) -> CRF:
    """
    构造线性链 CRF 对象。

    参数：
        algorithm='lbfgs'：用 L-BFGS 优化条件对数似然（带正则）。
        c1：L1 正则强度，促进稀疏权重（特征选择）。
        c2：L2 正则强度，抑制过大权重。
        all_possible_transitions：显式考虑所有标签对作为转移特征，未见过的转移在训练中惩罚为 0，解码时仍合法。
    """
    return CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=all_possible_transitions,
        verbose=False,
    )


def train_and_decode_validation(
    language: str,
    ner_root: Path | None = None,
    out_dir: Path | None = None,
    c1: float = 0.08,
    c2: float = 0.4,
    max_iterations: int = 80,
    save_model_path: Path | None = None,
) -> tuple[Path, Path, CRF]:
    """
    完整管线：读训练 → 提特征 build_xy → crf.fit → 读验证 token → crf.predict（维特比）→ 写 pred_path。
    CRF 训练与 sklearn-crfsuite 内部实现均为 CPU 密集型。
    Returns:
        (预测文件路径, 验证集 gold 路径（用于 check.py）, 训练好的 CRF 对象)
    """
    ner_root = ner_root or ner_data_dir()
    out_dir = out_dir or (ner_root / "predictions" / "crf")
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = ner_root / language
    train_path = sub / "train.txt"
    val_path = sub / "validation.txt"
    pred_path = out_dir / f"{language}_validation_crf.txt"

    print(f"[{language}] 读取训练集: {train_path}")
    train_sents = read_tagged_corpus(train_path)
    print(f"[{language}] 训练句数: {len(train_sents)}")

    print(f"[{language}] 构造特征序列...")
    X_train, y_train = build_xy(train_sents, language)

    print(f"[{language}] 训练 CRF（L-BFGS）...")
    crf = make_crf(c1=c1, c2=c2, max_iterations=max_iterations)
    # fit：估计每个特征在每个标签上的权重 + 标签转移权重
    # 优化目标为正则化对数似然
    crf.fit(X_train, y_train)

    print(f"[{language}] 读取验证集: {val_path}")
    val_tokens = read_tokens_only(val_path)
    if language == "English":
        fe = sentence_to_features_english
    else:
        fe = sentence_to_features_chinese

    # 验证阶段仅需要 X；标签由维特比解码得到
    X_val = [fe(s) for s in val_tokens]
    print(f"[{language}] 维特比解码验证集...")
    y_pred = crf.predict(X_val)

    print(f"[{language}] 写出预测: {pred_path}")
    write_predictions(pred_path, val_tokens, y_pred)

    # 写结果、存模型
    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        with save_model_path.open("wb") as f:
            pickle.dump({"language": language, "crf": crf}, f)
        print(f"[{language}] 模型已保存: {save_model_path}")

    return pred_path, val_path, crf


def print_check_hint(language: str, gold_path: Path, pred_path: Path) -> None:
    """打印如何用 NER/check.py 评测。"""
    gold_rel = os.path.relpath(gold_path, ner_data_dir())
    pred_rel = os.path.relpath(pred_path, ner_data_dir())
    print("\n--- 使用 NER/check.py 评测（在 NER 目录下执行）---")
    print(
        "python -c \"from check import check; "
        f"check(language='{language}', gold_path=r'{gold_rel}', my_path=r'{pred_rel}')\""
    )


def predict_with_model(
    model_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """加载 pickle 模型，对 input_path 解码并写出。"""
    with Path(model_path).open("rb") as f:
        bundle = pickle.load(f)
    lang = bundle["language"]
    crf: CRF = bundle["crf"]
    sents = read_tokens_only(input_path)
    fe = sentence_to_features_english if lang == "English" else sentence_to_features_chinese
    y_pred = crf.predict([fe(s) for s in sents])
    write_predictions(output_path, sents, y_pred)
    print(f"已写出: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="任务二：线性链 CRF 做 NER（sklearn-crfsuite）")
    parser.add_argument(
        "--lang",
        choices=["English", "Chinese", "both"],
        default="both",
        help="训练/解码的语言子集",
    )
    parser.add_argument("--c1", type=float, default=0.08, help="L1 正则系数")
    parser.add_argument("--c2", type=float, default=0.4, help="L2 正则系数")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=80,
        help="L-BFGS 最大迭代次数（英文语料较大时可酌情调高）",
    )
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="",
        help="若非空，写入 {English|Chinese}_crf.pkl",
    )
    parser.add_argument("--model-path", type=str, default="", help="仅解码：模型 pickle")
    parser.add_argument("--input-path", type=str, default="", help="仅解码：输入文件")
    parser.add_argument("--output-path", type=str, default="", help="仅解码：输出文件")
    args = parser.parse_args()

    if args.model_path or args.input_path or args.output_path:
        if not (args.model_path and args.input_path and args.output_path):
            raise SystemExit("仅解码需同时提供 --model-path --input-path --output-path")
        predict_with_model(Path(args.model_path), Path(args.input_path), Path(args.output_path))
        return

    langs = ["English", "Chinese"] if args.lang == "both" else [args.lang]
    save_dir = Path(args.save_model_dir) if args.save_model_dir else None

    for lang in langs:
        model_p = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            model_p = save_dir / f"{lang}_crf.pkl"
        pred_path, gold_path, _ = train_and_decode_validation(
            language=lang,
            c1=args.c1,
            c2=args.c2,
            max_iterations=args.max_iterations,
            save_model_path=model_p,
        )
        print_check_hint(lang, gold_path, pred_path)


if __name__ == "__main__":
    main()
