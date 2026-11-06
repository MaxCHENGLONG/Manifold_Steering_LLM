import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from manifold_guard.corpus import (Sequence, batches, collate, iter_chat, iter_fineweb, split_of,
                                   split_rows, take_budget)


def _bool(rows):
    return torch.tensor(rows, dtype=torch.bool)


def test_split_is_deterministic_and_balanced():
    names = [split_of("fineweb", f"doc{i}") for i in range(20000)]
    assert names == [split_of("fineweb", f"doc{i}") for i in range(20000)]
    for split, frac in (("fit", 0.90), ("calib", 0.05), ("holdout", 0.05)):
        assert abs(names.count(split) / len(names) - frac) < 0.01
    # 来源也参与哈希：同一个 id 在两个来源里各自切分
    assert [split_of("chat", f"doc{i}") for i in range(200)] != names[:200]


def test_collate_roles():
    batch = [Sequence("fineweb", "a", [5, 6, 7], 3), Sequence("chat", "b", [1, 2, 3, 4, 8], 3)]
    ids, am, roles = collate(batch, pad_id=0)
    assert ids.tolist() == [[5, 6, 7, 0, 0], [1, 2, 3, 4, 8]]
    assert am.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert torch.equal(roles["text"], _bool([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]))
    assert torch.equal(roles["prompt"], _bool([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0]]))
    assert torch.equal(roles["response"], _bool([[0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]))


def test_split_rows():
    batch = [Sequence("fineweb", f"d{i}", [1, 2], 2) for i in range(50)]
    for sp in ("fit", "calib", "holdout"):
        assert split_rows(batch, sp).view(-1).tolist() == [s.split == sp for s in batch]


def test_batches_respect_budget_and_keep_everything():
    g = torch.Generator().manual_seed(0)
    seqs = [Sequence("chat", f"c{i}", [1] * int(torch.randint(1, 50, (1,), generator=g)), 1) for i in range(300)]
    out = list(batches(seqs, max_tokens=200, buffer=64))
    assert sorted(id(s) for b in out for s in b) == sorted(id(s) for s in seqs)
    assert all(len(b) * max(len(s.input_ids) for s in b) <= 200 for b in out)
    with pytest.raises(ValueError):
        list(batches([Sequence("chat", "x", [1] * 300, 1)], max_tokens=200))


def test_take_budget_skips_holdout_and_stops_at_budget():
    seqs = [Sequence("fineweb", f"d{i}", [1] * 11, 11) for i in range(500)]  # 每条 10 个参与统计的 token
    taken = list(take_budget(seqs, 1000))
    assert all(s.split != "holdout" for s in taken)
    assert sum(len(s.input_ids) - 1 for s in taken if s.split == "fit") == 1000
    assert any(s.split == "calib" for s in taken)


class _CharTokenizer:
    """按字符分词的假 tokenizer，接口与 HF tokenizer 的批量调用一致。"""

    def __call__(self, texts, truncation, max_length):
        return {"input_ids": [[ord(c) for c in t][:max_length] for t in texts]}


def test_iter_fineweb(tmp_path):
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"id": ["a", "b", "c"], "text": ["hello world", "x", "abcdefgh"],
                             "url": ["u"] * 3}), path)
    seqs = list(iter_fineweb(path, _CharTokenizer(), max_len=5))
    assert [s.doc_id for s in seqs] == ["a", "c"]  # 只有 1 个 token 的文档跳过
    assert seqs[0].input_ids == [ord(c) for c in "hello"]
    assert all(s.source == "fineweb" and s.prompt_len == len(s.input_ids) for s in seqs)


def test_iter_chat(tmp_path):
    path = tmp_path / "chat.jsonl"
    rows = [{"id": "p1", "prompt_token_ids": [1, 2, 3], "response_token_ids": [4, 5]},
            {"id": "p2", "prompt_token_ids": [1, 2, 3, 4, 5, 6], "response_token_ids": [7]}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    a, b = iter_chat(path, max_len=4)
    assert (a.input_ids, a.prompt_len) == ([1, 2, 3, 4], 3)
    assert (b.input_ids, b.prompt_len) == ([1, 2, 3, 4], 4)  # 截断后整条都是 prompt
