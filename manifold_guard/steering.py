"""激活 steering 的挂法与系数换算，以及 TinyMMLU 的答案解析。

向量格式与 persona_vectors 相同：形状 [L+1, d]，第 0 行是 embedding 层，`vector[l]` 对应
`model.layers[l-1]` 的输出。steering 层号沿用 weight-steering 的约定（1.5B 默认 17）。
"""
import copy
import re
from contextlib import nullcontext

import torch

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
