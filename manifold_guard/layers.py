"""定位解码层，在层输出上挂 forward hook 取出 hidden state，以及选出参与统计的 token。

层号约定：本包里的层号 l 一律指 `model.layers[l]` 的输出，与 ActivationSteerer 的
layer_idx 相同。persona_vectors 的向量把第 0 行留给 embedding 层，所以 `vector[l]`
对应本包的第 l-1 层（见 weight-steering 的 select_steerer）。

hook 顺序：同一层的 forward hook 按注册顺序执行。LayerTap 要看到 steering 之后的值，
就必须在 steerer 之后进入，即 `with steerer, LayerTap(...)`。

不用 `output_hidden_states`：HF 返回的最后一项已经过 final norm，和 steerer 挂的位置不一致。
"""
from types import SimpleNamespace

import torch

from manifold_guard.upstream import activation_steering

_ActivationSteerer = activation_steering().ActivationSteerer


def num_layers(model):
    return model.config.num_hidden_layers


def locate_layer(model, layer_idx):
    """返回 `model.layers[layer_idx]` 对应的模块。

    直接调用 ActivationSteerer._locate_layer，保证 LayerTap、guard 和 steerer 挂在同一个模块上。
    """
    probe = SimpleNamespace(model=model, layer_idx=layer_idx, debug=False,
                            _POSSIBLE_LAYER_ATTRS=_ActivationSteerer._POSSIBLE_LAYER_ATTRS)
    return _ActivationSteerer._locate_layer(probe)


def unwrap(output):
    """解码层的输出可能是张量，也可能是 (hidden_states, ...) 元组。"""
    return output[0] if isinstance(output, (tuple, list)) else output


class LayerTap:
    """上下文管理器：在指定层挂 forward hook，缓存输出（兼容 tuple）。

    outputs[l] 是最近一次前向里第 l 层的输出，形状 (B, T, d)；positions="last" 时只留
    最后一个位置，形状 (B, 1, d)。默认 detach；训练时的正则（M4）需要梯度，传 detach=False。
    hook 只读不写，不改变模型输出。
    """

    def __init__(self, model, layers="all", positions="all", detach=True):
        if positions not in ("all", "last"):
            raise ValueError("positions 只能是 'all' 或 'last'")
        n = num_layers(model)
        if layers == "all":
            layers = range(n)
        for l in layers:
            if not -n <= l < n:
                raise IndexError(f"层号 {l} 超出范围（模型共 {n} 层）")
        self.model = model
        self.layers = sorted({l % n for l in layers})
        self.positions = positions
        self.detach = detach
        self.outputs: dict[int, torch.Tensor] = {}
        self._handles = []

    def _hook(self, layer):
        def hook(module, inputs, output):
            h = unwrap(output)
            if self.positions == "last":
                h = h[:, -1:, :]
            self.outputs[layer] = h.detach() if self.detach else h
        return hook

    def __enter__(self):
        self.outputs = {}
        for l in self.layers:
            self._handles.append(locate_layer(self.model, l).register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def token_mask(attention_mask, drop_first=True):
    """参与统计的 token：去掉 padding，默认再去掉每条序列的第一个有效 token。

    第一个 token 是 attention sink，范数比其他位置大几个数量级，会把协方差带偏。
    左右 padding 都适用。返回 (B, T) 的布尔张量。
    """
    keep = attention_mask.bool()
    if not drop_first:
        return keep
    rank = attention_mask.long().cumsum(dim=1)  # 有效 token 在本序列里的序号，从 1 开始
    return keep & (rank > 1)


def role_masks(attention_mask, prompt_lens, drop_first=True):
    """把参与统计的 token 分成 prompt 和 response 两部分。

    prompt_lens[i] 是第 i 条序列 prompt 部分的 token 数（不含 padding，含首 token）。
    返回 {"prompt": (B, T), "response": (B, T)} 两个布尔张量。
    """
    keep = token_mask(attention_mask, drop_first)
    rank = attention_mask.long().cumsum(dim=1)
    p = torch.as_tensor(prompt_lens, device=attention_mask.device).view(-1, 1)
    return {"prompt": keep & (rank <= p), "response": keep & (rank > p)}
