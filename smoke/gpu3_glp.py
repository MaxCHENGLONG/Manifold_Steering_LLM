"""单卡冒烟测试 3：加载多层 GLP（Llama-3.2-1B 第 0–15 层），在几个层上各采样 64 条激活。

GLP 的代码来自子模块 third_party/generative_latent_prior，权重已预下载到 $HF_HOME。
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
GLP_SRC = REPO / "third_party" / "generative_latent_prior"
if not GLP_SRC.is_dir():
    sys.exit(f"找不到 GLP 代码：{GLP_SRC}\n请先在 $REPO 里添加子模块（见说明）。")
sys.path.insert(0, str(GLP_SRC))

from glp import flow_matching  # noqa: E402
from glp.denoiser import load_glp  # noqa: E402


def main():
    device = "cuda:0"
    model = load_glp("generative-latent-prior/glp-llama1b-d12-multi", device=device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    d = model.denoiser.model.d_input
    print(f"  GLP 已加载：{n_params:.2f}B 参数，激活维度 {d}")

    noise = torch.randn(64, 1, d, device=device)
    for layer in (0, 7, 15):
        latents = flow_matching.sample(model, noise, num_timesteps=20, layer_idx=layer)
        acts = model.normalizer.denormalize(latents, layer_idx=layer)
        assert torch.isfinite(acts).all(), f"第 {layer} 层采样出现 NaN/Inf"
        print(f"  第 {layer:2d} 层：形状 {tuple(acts.shape)}，平均范数 {acts.float().norm(dim=-1).mean():.2f}")

    print("GPU3 GLP PASS")


if __name__ == "__main__":
    main()
