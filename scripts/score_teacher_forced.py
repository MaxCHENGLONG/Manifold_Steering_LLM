#!/usr/bin/env python3
"""C2：把 steering 后生成的回答（prompt + 回答）一次喂回模型，逐层给 response token 打分。

两种条件：
  steered     喂给生成它的那个被 steering 的模型（激活 steering 用 MaskedSteerer，只加在生成时会加的位置）。主测量。
  base_model  喂给原模型。对照：分数高是因为文本本身不自然，还是因为 steering 后模型的内部计算不自然。
基线配置（k=0）两种条件相同，只算 steered。
每条回答、每层、每个打分器记录 response token 数、z 的均值和最大值、超过校准 99% / 99.9% 分位数的比例。
轨迹分数（每个 token 取各层 z 的最大值）的同样统计记在 layer = -1，阈值用它自己在 calib 样本上的分位数。

输入两种（--kind）：
  generated  generate_steered.py 的 responses.jsonl，有 token id，原样喂回。
  tinymmlu   sweep_capability.py 的 responses.jsonl，没有 token id：按数据集里的对话重新套模板，回答文本重新分词，
             没写到上限的回答补上结束符。
配置（方法、强度、绝对系数）取自粗扫的 summary.json。

  python scripts/score_teacher_forced.py --model Qwen/Qwen2.5-1.5B-Instruct --kind generated \\
      --responses results/Qwen2.5-1.5B-Instruct/c2/sycophancy --coarse results/Qwen2.5-1.5B-Instruct/coarse \\
      --stats stats/Qwen2.5-1.5B-Instruct --delta ckpt/Qwen2.5-1.5B-Instruct/sycophancy/delta.pt \\
      --vector data/vectors/Qwen2.5-1.5B-Instruct/sycophantic_response_avg_diff.pt
输出 <responses>/<配置名>/scores.parquet；已存在则跳过（--overwrite 重算）。
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.calibrate import calib_path, calibrate, save_calibration, standardize, trajectory_score  # noqa: E402
from manifold_guard.corpus import Sequence, batches, collate  # noqa: E402
from manifold_guard.hfdata import load_split  # noqa: E402
from manifold_guard.layers import LayerTap  # noqa: E402
from manifold_guard.scorers import make_scorer  # noqa: E402
from manifold_guard.steering import MaskedSteerer, steer_mask  # noqa: E402
from manifold_guard.weight_steering import WeightSteerer  # noqa: E402

QS = ("0.99", "0.999")
COLUMNS = ("example", "layer", "scorer", "n_tokens", "z_mean", "z_max", *(f"frac_q{q[2:]}" for q in QS))


def load_generated(path):
    with open(path) as f:
        recs = [json.loads(line) for line in f]
    return [Sequence("chat", str(r["example"]), r["prompt_ids"] + r["response_ids"], len(r["prompt_ids"]))
            for r in recs]


def load_tinymmlu(path, tok, rows, max_new_tokens):
    """重建 sweep_capability.py 生成时的 token：prompt 与 steering.generate 的套模板方式相同。"""
    with open(path) as f:
        recs = [json.loads(line) for line in f]
    if len(recs) != len(rows):
        raise ValueError(f"{path} 有 {len(recs)} 条，数据集有 {len(rows)} 条")
    seqs = []
    for k, (r, row) in enumerate(zip(recs, rows)):
        text = tok.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)
        p_ids = tok(text, add_special_tokens=False)["input_ids"]
        r_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        if r["new_tokens"] < max_new_tokens:
            r_ids.append(tok.eos_token_id)
        seqs.append(Sequence("chat", str(k), p_ids + r_ids, len(p_ids)))
    return seqs


def _aggregate(out, ex, layer, name, z, thresholds):
    """把一个 batch 里各 token 的 z 按回答汇总，追加到 out（列 → 列表）。"""
    uniq, inv = torch.unique(ex.to(z.device), return_inverse=True)
    n = torch.bincount(inv, minlength=len(uniq))
    zf = z.double()
    total = torch.zeros(len(uniq), dtype=torch.float64, device=z.device).index_add_(0, inv, zf)
    zmax = torch.full((len(uniq),), -torch.inf, dtype=torch.float64, device=z.device).scatter_reduce_(
        0, inv, zf, "amax")
    over = {q: torch.zeros(len(uniq), dtype=torch.float64, device=z.device).index_add_(
        0, inv, (zf > thresholds[q]).double()) / n for q in QS}
    out["example"] += uniq.tolist()
    out["layer"] += [layer] * len(uniq)
    out["scorer"] += [name] * len(uniq)
    out["n_tokens"] += n.tolist()
    out["z_mean"] += (total / n).tolist()
    out["z_max"] += zmax.tolist()
    for q in QS:
        out[f"frac_q{q[2:]}"] += over[q].tolist()


@torch.no_grad()
def score_records(model, seqs, scorers, calibs, layers, pad_id, batch_tokens, steerer=None):
    """返回 {列名: 列表}，每个 (回答, 层, 打分器) 一行；回答号取自 Sequence.doc_id。"""
    out = {c: [] for c in COLUMNS}
    dev = next(model.parameters()).device
    with LayerTap(model, layers) as tap:
        for b in batches(seqs, batch_tokens):
            ids, am, roles = collate(b, pad_id)
            ids, am, resp = ids.to(dev), am.to(dev), roles["response"].to(dev)
            if steerer is not None:
                steerer.mask = steer_mask(am, [s.prompt_len for s in b])
            model(input_ids=ids, attention_mask=am)
            ex = torch.tensor([int(s.doc_id) for s in b], device=dev).view(-1, 1).expand_as(resp)[resp]
            for sc in scorers:
                c = calibs[sc.name]
                z = {l: standardize(sc.score(tap.outputs[l][resp], l), c, l) for l in layers}
                for l in layers:
                    _aggregate(out, ex, l, sc.name, z[l], c["layers"][l]["z_quantiles"])
                _aggregate(out, ex, -1, sc.name, trajectory_score(z, "max"), c["trajectory"]["max"]["quantiles"])
    return out


def main():
    ap = argparse.ArgumentParser(description="C2：teacher-forcing 逐层打分")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=("generated", "tinymmlu"), required=True)
    ap.add_argument("--responses", help="各配置子目录所在的目录；tinymmlu 默认就是 --coarse")
    ap.add_argument("--coarse", required=True, help="sweep_capability.py 的输出目录（取配置）")
    ap.add_argument("--stats", required=True, help="模型的逐层统计目录（含各角色子目录）")
    ap.add_argument("--role", default="response")
    ap.add_argument("--scorers", default="norm,gauss,pca:var=0.9,pca:var=0.95,pca:var=0.99")
    ap.add_argument("--delta", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--layer", type=int, default=17, help="激活 steering 层（weight-steering 约定）")
    ap.add_argument("--methods", default="base,weight,act,act_all")
    ap.add_argument("--configs", help="只算这些配置（逗号分隔）")
    ap.add_argument("--conditions", default="steered,base_model")
    ap.add_argument("--batch_tokens", type=int, default=32768)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    t_all = time.time()
    summary = json.loads((Path(args.coarse) / "summary.json").read_text())
    coarse = summary["configs"]
    resp_dir = Path(args.responses or args.coarse)
    names = args.configs and args.configs.split(",")
    configs = {n: m for n, m in coarse.items() if m["method"] in args.methods.split(",")
               and (names is None or n in names) and (resp_dir / n / "responses.jsonl").exists()}
    todo = {n: m for n, m in configs.items() if args.overwrite or not (resp_dir / n / "scores.parquet").exists()}
    print(f"{len(configs)} 个配置有回答，{len(todo)} 个要打分", flush=True)
    if not todo:
        return

    role_dir = Path(args.stats) / args.role
    scorers = [make_scorer(s).load(role_dir) for s in args.scorers.split(",")]
    layers = scorers[0].layers
    calibs = {}
    for sc in scorers:
        calibs[sc.name] = calibrate(sc, role_dir)
        save_calibration(calibs[sc.name], calib_path(role_dir, sc.name))

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    delta = {k: v.to("cuda") for k, v in torch.load(args.delta).items()}
    vector = torch.load(args.vector)
    if args.kind == "tinymmlu":
        sargs = summary["args"]
        rows = load_split(sargs["dataset"], sargs["split"])[: sargs.get("limit")]

    for name, m in todo.items():
        t0 = time.time()
        path = resp_dir / name / "responses.jsonl"
        seqs = load_generated(path) if args.kind == "generated" else load_tinymmlu(
            path, tok, rows, summary["args"]["max_new_tokens"])
        cols = {c: [] for c in ("condition", *COLUMNS)}
        for cond in args.conditions.split(","):
            if cond == "base_model" and m["method"] == "base":
                continue
            steerer, ctx = None, nullcontext()
            if cond == "steered" and m["method"] == "weight":
                ctx = WeightSteerer(model, delta, m["strength"])
            elif cond == "steered" and m["method"] in ("act", "act_all"):
                ctx = steerer = MaskedSteerer(model, vector, m["coeff"], args.layer if m["method"] == "act" else 0)
            with ctx:
                res = score_records(model, seqs, scorers, calibs, layers, pad_id, args.batch_tokens, steerer)
            cols["condition"] += [cond] * len(res["example"])
            for c in COLUMNS:
                cols[c] += res[c]
        table = pa.table({"config": [name] * len(cols["example"]), **cols})
        pq.write_table(table, resp_dir / name / "scores.parquet")
        # 终端里给一行概况：各条件下轨迹分数超过 99% 分位数的 token 比例（按回答平均），取前两个便宜打分器
        show = [s.name for s in scorers if s.name != "norm"][:2]
        brief = []
        for cond in dict.fromkeys(cols["condition"]):
            for s in show:
                v = [f for c, l, sc, f in zip(cols["condition"], cols["layer"], cols["scorer"], cols["frac_q99"])
                     if c == cond and l == -1 and sc == s]
                brief.append(f"{cond}/{s} {sum(v) / len(v):.3f}")
        print(f"{name:16s} {len(seqs)} 条  " + "  ".join(brief) + f"  {time.time() - t0:.0f} s", flush=True)
    print(f"共 {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
