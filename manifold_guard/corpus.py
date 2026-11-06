"""逐层统计用的自然语料：读取、按文档切分、打包成 batch。

两类来源，对应三个角色：
  fineweb  纯文本文档，全部 token 记为 text 角色
  chat     UltraChat 的首轮问题加被测模型自己生成的回答（on-policy）：prompt 部分
           （含聊天模板）记为 prompt 角色，回答部分记为 response 角色

文档按 id 的哈希固定地分到 fit（90%）、calib（5%）、holdout（5%）三份，与读取顺序、
分块方式和机器都无关：fit 用来累计均值和协方差，calib 用来校准阈值，holdout 留给恒等性检查。
"""
import hashlib
import json
from dataclasses import dataclass, field

import pyarrow.parquet as pq
import torch

from manifold_guard.layers import role_masks

ROLES = ("text", "prompt", "response")
SPLITS = (("fit", 90), ("calib", 5), ("holdout", 5))


def split_of(source, doc_id):
    h = int.from_bytes(hashlib.sha256(f"{source}:{doc_id}".encode()).digest()[:8], "big") % 100
    for name, pct in SPLITS:
        if h < pct:
            return name
        h -= pct


@dataclass
class Sequence:
    source: str           # "fineweb" | "chat"
    doc_id: str
    input_ids: list[int]
    prompt_len: int       # chat：prompt 部分的 token 数；fineweb：等于全长
    split: str = field(init=False)

    def __post_init__(self):
        self.split = split_of(self.source, self.doc_id)


def iter_fineweb(path, tokenizer, max_len):
    """逐条读取 FineWeb 分片（parquet），分词后截断到 max_len。"""
    for rb in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["id", "text"]):
        rows = rb.to_pydict()
        enc = tokenizer(rows["text"], truncation=True, max_length=max_len)["input_ids"]
        for doc_id, ids in zip(rows["id"], enc):
            if len(ids) >= 2:
                yield Sequence("fineweb", doc_id, ids, len(ids))


def iter_chat(path, max_len):
    """读取 generate_onpolicy_chat.py 的输出（jsonl）。直接用生成时的 token id，不重新分词。"""
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ids = (r["prompt_token_ids"] + r["response_token_ids"])[:max_len]
            yield Sequence("chat", r["id"], ids, min(len(r["prompt_token_ids"]), len(ids)))


def take_budget(seqs, budget):
    """跳过 holdout，在 fit 切分累计到 budget 个参与统计的 token 后停止（calib 随之按比例读入）。"""
    fit = 0
    for s in seqs:
        if s.split == "holdout":
            continue
        if fit >= budget:
            return
        if s.split == "fit":
            fit += len(s.input_ids) - 1  # 首 token 不参与统计
        yield s


def batches(seqs, max_tokens, buffer=4096):
    """每攒 buffer 条序列按长度排序再打包，每个 batch 的 B × 最大长度不超过 max_tokens，padding 少。"""
    buf = []
    for s in seqs:
        if len(s.input_ids) > max_tokens:
            raise ValueError(f"序列长 {len(s.input_ids)}，超过 max_tokens={max_tokens}")
        buf.append(s)
        if len(buf) == buffer:
            yield from _pack(buf, max_tokens)
            buf = []
    yield from _pack(buf, max_tokens)


def _pack(buf, max_tokens):
    batch = []
    for s in sorted(buf, key=lambda s: len(s.input_ids)):
        if batch and (len(batch) + 1) * len(s.input_ids) > max_tokens:
            yield batch
            batch = []
        batch.append(s)
    if batch:
        yield batch


def collate(batch, pad_id):
    """右侧 padding，返回 input_ids、attention_mask 和各角色的 token 掩码（均为 (B, T)）。

    用右侧 padding：不传 position_ids 时，HF 按 0..T-1 编位置，左侧 padding 会让有效 token 的位置整体后移。
    """
    T = max(len(s.input_ids) for s in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), T), dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, : len(s.input_ids)] = torch.tensor(s.input_ids)
        am[i, : len(s.input_ids)] = 1
    m = role_masks(am, [s.prompt_len for s in batch])
    is_text = torch.tensor([s.source == "fineweb" for s in batch]).view(-1, 1)
    roles = {"text": m["prompt"] & is_text, "prompt": m["prompt"] & ~is_text, "response": m["response"]}
    return ids, am, roles


def split_rows(batch, split):
    """(B, 1) 的布尔张量：每条序列是否属于 split。"""
    return torch.tensor([s.split == split for s in batch]).view(-1, 1)
