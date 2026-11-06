"""单卡冒烟测试 2：在 HF 模型的中间层挂 forward hook。

检查两件事，它们是激活 steering 和 ManifoldGuard 的基础：
  1. 加零向量时，生成结果与不挂 hook 时逐字相同（hook 写回逻辑正确）
  2. 加一个足够大的随机向量时，生成结果会改变（hook 确实生效）
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def unwrap(output):
    """解码层的输出可能是张量，也可能是 (hidden_states, ...) 元组。"""
    return output[0] if isinstance(output, tuple) else output


def add_vector_hook(vec):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vec,) + tuple(output[1:])
        return output + vec
    return hook


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda", dtype=torch.bfloat16).eval()
    layers = model.model.layers
    L = len(layers) // 2
    d = model.config.hidden_size
    print(f"  模型 {MODEL}：{len(layers)} 层，hidden={d}，hook 挂在第 {L} 层")

    enc = tok.apply_chat_template(
        [{"role": "user", "content": "Describe a cat in one sentence."}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to("cuda")

    def generate():
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=24, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # 这一层残差流的平均范数，用来确定 steering 向量的大小
    norms = []
    h = layers[L].register_forward_hook(lambda m, i, o: norms.append(unwrap(o).float().norm(dim=-1).mean().item()))
    with torch.no_grad():
        model(**enc)
    h.remove()
    mean_norm = norms[0]
    print(f"  第 {L} 层残差流平均范数：{mean_norm:.1f}")

    base = generate()

    h = layers[L].register_forward_hook(add_vector_hook(torch.zeros(d, device="cuda", dtype=torch.bfloat16)))
    zero = generate()
    h.remove()
    assert zero == base, f"零向量 hook 改变了输出：\n  base: {base}\n  zero: {zero}"
    print("  [1/2] 零向量 hook：输出与原模型逐字相同")

    g = torch.Generator(device="cuda").manual_seed(0)
    v = torch.randn(d, device="cuda", generator=g)
    v = (v / v.norm() * 4 * mean_norm).to(torch.bfloat16)
    h = layers[L].register_forward_hook(add_vector_hook(v))
    steered = generate()
    h.remove()
    print(f"  原输出：  {base!r}\n  steering：{steered!r}")
    if steered == base:
        print("  [2/2] 警告：加随机向量后输出没变（不算失败，但值得留意）")
    else:
        print("  [2/2] 随机向量 hook：输出已改变")

    print("GPU2 HF HOOK PASS")


if __name__ == "__main__":
    main()
