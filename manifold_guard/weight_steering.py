"""权重 steering：θ_k = θ_base + k·(θ⁺ − θ⁻)。

与 weight-steering 的 task_vectors.py 公式相同（model1 取基座本身，scale_t2 = scale_t3 = k），
实现上有两点不同：
  - Δ = θ⁺ − θ⁻ 在 float32 里算一次存下来。lr 1e-5 训练出的差很小，合并后的模型若以 bf16 保存，
    这个差会被量化掉。
  - 扫描时用 WeightSteerer 在显存里给参数加上 kΔ，退出时恢复，不为每个 k 落盘。
参数名按 named_parameters 去重：共享的 embed_tokens / lm_head 只出现一次。
"""
import torch


def merged_params(base_model, adapter_dir):
    """float32 加载基座、套上 LoRA 适配器并合并，返回 {参数名: 张量}。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    return {n: p.detach() for n, p in model.named_parameters()}


def compute_delta(base_model, pos_adapter, neg_adapter):
    """Δ = θ⁺ − θ⁻（float32），只保留真正有变化的参数。"""
    pos = merged_params(base_model, pos_adapter)
    neg = merged_params(base_model, neg_adapter)
    if pos.keys() != neg.keys():
        raise ValueError("两个适配器合并后的参数名不一致")
    delta = {n: pos[n] - neg[n] for n in pos}
    return {n: d for n, d in delta.items() if d.abs().max() > 0}


class WeightSteerer:
    """上下文管理器：把参数临时改成 θ + k·Δ，退出时恢复原值（k=0 时与原模型逐位相同）。"""

    def __init__(self, model, delta, k):
        self.params = dict(model.named_parameters())
        if missing := set(delta) - set(self.params):
            raise KeyError(f"模型里没有这些参数：{sorted(missing)[:5]}")
        self.delta, self.k = delta, float(k)
        self._saved = {}

    @torch.no_grad()
    def __enter__(self):
        for n, d in self.delta.items():
            p = self.params[n]
            self._saved[n] = p.detach().clone()
            p.copy_((p.float() + self.k * d.to(p.device, torch.float32)).to(p.dtype))
        return self

    @torch.no_grad()
    def __exit__(self, *exc):
        for n, saved in self._saved.items():
            self.params[n].copy_(saved)
        self._saved.clear()
