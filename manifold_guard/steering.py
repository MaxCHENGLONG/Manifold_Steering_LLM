"""激活 steering 的挂法与系数换算，以及 TinyMMLU 的答案解析。

向量格式与 persona_vectors 相同：形状 [L+1, d]，第 0 行是 embedding 层，`vector[l]` 对应
`model.layers[l-1]` 的输出。steering 层号沿用 weight-steering 的约定（1.5B 默认 17）。
"""
import copy
import re
from contextlib import nullcontext

import torch

from manifold_guard.layers import locate_layer, unwrap
from manifold_guard.upstream import activation_steering

_steering = activation_steering()


def act_steerer(model, vector, coeff, layer, positions="response"):
    """与 weight-steering 的 select_steerer 相同。

    layer ≥ 1：在 model.layers[layer-1] 上加 coeff·vector[layer]。
    layer == 0（全层版本）：对 l = 1..L-1，在 model.layers[l-1] 上加 coeff·(vector[l] − vector[l-1])
    （l = 1 时为 vector[1]），使第 l 层输出处的累计偏移约为 coeff·vector[l]。
    """
    if coeff == 0:
        return nullcontext()
    if layer >= 1:
        return _steering.ActivationSteerer(model, vector[layer], coeff=coeff, layer_idx=layer - 1,
                                           positions=positions)
    return _steering.ActivationSteererMultiple(model, [
        dict(steering_vector=vector[l] - vector[l - 1] if l > 1 else vector[l], coeff=coeff,
             layer_idx=l - 1, positions=positions)
        for l in range(1, model.config.num_hidden_layers)
    ])


def steer_mask(attention_mask, prompt_lens):
    """teacher-forcing 时要加 steering 的位置：从最后一个 prompt token 起的全部有效 token（右侧 padding）。

    生成时 positions="response" 每步只改最后一个位置：prefill 时是最后一个 prompt token，之后是每个
    新喂入的回答 token。把"prompt + 回答"一次喂进去时，被改的正是这些位置（多出的最后一个回答
    token 只影响它自己的输出）。
    """
    rank = attention_mask.long().cumsum(dim=1)
    p = torch.as_tensor(prompt_lens, device=attention_mask.device).view(-1, 1)
    return attention_mask.bool() & (rank >= p)


class MaskedSteerer:
    """和 act_steerer 加同样的向量、挂同样的层，但只加在 mask 为 True 的位置。

    用于把整段文本一次喂进被 steering 的模型（teacher forcing）。每次前向前设置 .mask，形状 (B, T)。
    向量先转成模型的 dtype 再乘系数，与上游 ActivationSteerer 的数值一致。
    """

    def __init__(self, model, vector, coeff, layer):
        if layer >= 1:
            adds = [(layer - 1, vector[layer])]
        else:
            adds = [(l - 1, vector[l] - vector[l - 1] if l > 1 else vector[l])
                    for l in range(1, model.config.num_hidden_layers)]
        p = next(model.parameters())
        self.adds = [(i, float(coeff) * v.to(p.device, p.dtype)) for i, v in adds]
        self.model, self.mask, self._handles = model, None, []

    def _hook(self, steer):
        def fn(module, ins, out):
            h = unwrap(out)
            h = h + steer * self.mask.unsqueeze(-1).to(h.dtype)
            return (h, *out[1:]) if isinstance(out, (tuple, list)) else h
        return fn

    def __enter__(self):
        self._handles = [locate_layer(self.model, i).register_forward_hook(self._hook(s)) for i, s in self.adds]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def absolute_coeff(c_rel, vector, layer, median_norm):
    """相对系数换成绝对系数：c_rel = c·‖vector[layer]‖ / median_norm。

    median_norm 是 model.layers[layer-1] 输出在自然 response token 上的范数中位数（逐层统计里的
    norm_quantiles[0.5]）。全层版本也按参考层 layer 换算，两种激活 steering 在该层的偏移大小相同。
    """
    return c_rel * median_norm / vector[layer].float().norm().item()


_FINAL_ANSWER = re.compile(r"final answer[*\s]*[:：]\s*[*(\[\s]*([A-D])(?![A-Za-z])", re.IGNORECASE)


def parse_choice(text):
    """取最后一个 "Final Answer: X" 里的选项字母；没有则返回 None。"""
    found = _FINAL_ANSWER.findall(text)
    return found[-1].upper() if found else None


def greedy_config(model, max_new_tokens):
    """贪心解码；repetition_penalty 等其余参数沿用模型的 generation_config，与上游用 vLLM 评测时一致。"""
    cfg = copy.deepcopy(model.generation_config)
    cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k = False, None, None, None
    cfg.max_new_tokens = max_new_tokens
    return cfg


@torch.no_grad()
def generate(model, tok, conversations, gen_config, batch_size):
    """左侧 padding 批量生成，返回每条的 (回答文本, 新 token 数)。"""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_config.pad_token_id = tok.pad_token_id
    texts = [tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i: i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        ids = model.generate(**enc, generation_config=gen_config)
        new = ids[:, enc["input_ids"].shape[1]:]
        for row in new:
            n = int((row != tok.pad_token_id).sum())
            out.append((tok.decode(row, skip_special_tokens=True), n))
    return out
