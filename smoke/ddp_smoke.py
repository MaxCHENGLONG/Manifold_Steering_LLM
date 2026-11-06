"""4 卡 DDP 冒烟测试。

检查三件事：
  1. NCCL 通信结果正确（all_reduce 求和）
  2. 卡间通信带宽（1 GiB bf16 all_reduce）
  3. 小模型 + LoRA 的 DDP 训练能跑通，且训练后各卡参数一致

使用随机初始化的小 Qwen2 模型，不需要下载任何权重，也不需要外网。

用法（在容器的 train 环境里）：
  torchrun --standalone --nproc_per_node=4 ddp_smoke.py
"""
import os
import socket
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    cpus = sorted(os.sched_getaffinity(0))
    print(
        f"[rank {rank}] host={socket.gethostname()} gpu={torch.cuda.get_device_name(dev)} "
        f"local_rank={local} 可用CPU核={len(cpus)} ({cpus[0]}-{cpus[-1]})",
        flush=True,
    )
    dist.barrier()

    # ---------- 1. 通信正确性 ----------
    t = torch.full((1,), float(rank + 1), device=dev)
    dist.all_reduce(t)
    expected = world * (world + 1) / 2
    assert t.item() == expected, f"all_reduce 结果错误：{t.item()} != {expected}"
    if rank == 0:
        print(f"[1/3] all_reduce 正确：{t.item():.0f} == {expected:.0f}（world={world}）", flush=True)

    # ---------- 2. 通信带宽 ----------
    x = torch.zeros(512 * 1024 * 1024, dtype=torch.bfloat16, device=dev)  # 1 GiB
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    size = x.numel() * x.element_size()
    algbw = size / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world
    if rank == 0:
        print(
            f"[2/3] all_reduce 1 GiB：每次 {dt * 1e3:.1f} ms，"
            f"algbw {algbw:.0f} GB/s，busbw {busbw:.0f} GB/s",
            flush=True,
        )
    del x
    torch.cuda.empty_cache()

    # ---------- 3. LoRA + DDP 训练 ----------
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)  # 各卡初始权重相同（DDP 构造时也会从 rank 0 广播）
    cfg = Qwen2Config(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(cfg).to(dev, dtype=torch.bfloat16)
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = DDP(model, device_ids=[local])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    gen = torch.Generator().manual_seed(1000 + rank)  # 各卡用不同数据，检验梯度同步
    for step in range(5):
        ids = torch.randint(0, cfg.vocab_size, (8, 128), generator=gen).to(dev)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        mean_loss = loss.detach().float()
        dist.all_reduce(mean_loss)
        if rank == 0:
            print(f"      step {step}  平均 loss {mean_loss.item() / world:.4f}", flush=True)

    # 训练后各卡的 LoRA 参数必须完全一致，否则说明梯度没有同步
    flat = torch.cat([p.detach().float().flatten() for p in model.parameters() if p.requires_grad])
    checksum = flat.sum().reshape(1)
    gathered = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(gathered, checksum)
    vals = [v.item() for v in gathered]
    assert max(vals) - min(vals) <= 1e-4 * max(1.0, abs(vals[0])), f"各卡 LoRA 参数不一致：{vals}"
    if rank == 0:
        print(f"[3/3] LoRA DDP 训练 5 步完成，各卡参数一致（checksum {vals[0]:.4f}）", flush=True)

    dist.barrier()
    if rank == 0:
        print("DDP SMOKE TEST PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
