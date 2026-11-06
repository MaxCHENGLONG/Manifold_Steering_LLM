# 流形约束前向推理：代码实现计划

> 本文件导出自 Claude 文档《流形约束前向推理：代码实现计划》（https://claude.ai/artifact/7jtT9gQxwcqMkNPCgVDzKU），导出时间 2026-11-06。文档更新后重新导出并提交，以保持两边一致。原文档中的两张图在这里分别换成了 Mermaid 流程图和表格。

## 目标与范围

在 weight-steering 与 GLP 两个现有代码库之上新建一个 `manifold_guard` 包：先用它测量 steering 造成的离流形程度（H1/H2），再把它变成推理时和训练时的约束（M0–M4）。开发调试用 1 块 GH200，训练作业申请整个节点的 4 块；按一人全职约 3–4 个月。

| 实验线 | 模型 | 流形模型 | 直接复用 | 负责的问题 |
| --- | --- | --- | --- | --- |
| Qwen 线 | Qwen2.5-1.5B-Instruct（开发）、Qwen2.5-7B-Instruct（主实验） | 逐层高斯 / PCA | weight-steering 的数据、axolotl 配置、`task_vectors.py`、Claude judge 评测 | H1/H2 诊断、M2、M4 |
| Llama 线 | Llama-3.2-1B-Instruct、Llama-3.1-8B-Instruct | GLP（1B 多层第 0–15 层；8B 第 15 层）+ 高斯 | GLP 预训练权重、`sample_on_manifold`、persona_vectors 集成 | M0/M1/M2 全层约束、M3 |

总体成功判据（阈值是本计划自定的，可在阶段 0 后调整）：

- **恒等性**：自然输入下开启 guard，干净文本 CE loss 增量 ≤ 1%，门控触发率 ≤ 1%。
- **H1**：在 k 扫描中，离流形分数与能力指标（TinyMMLU、GSM8K）的 Spearman 相关 |ρ| ≥ 0.7。
- **H2**：在相同行为改变量下，weight steering 的离流形分数低于 activation steering。
- **方法**：M2 或 M3 在 sycophancy 和 evil 两个行为上，把"行为改变 vs 能力保持"的 Pareto 前沿往外推。

不在范围内：跨节点训练、为 Qwen 训练全规模（10 亿条激活）GLP、闭源模型。

## 当前进度与环境速查（2026-11-06）

截至 2026-11-06，Arrhenius 上的容器环境已经建好，并通过了单卡与 4 卡测试；代码仓库已配置好 GitHub 的 SSH 认证；`smoke`、`qwen`、`llama`、`data` 四组模型和数据集已下载到 `$HF_HOME`（可选的 `refusal` 组未下载）。本节记录实际情况：下文原计划里的目录名、镜像名和 venv 切换方式如果与本节不一致，以本节为准。

### 已完成与待办

- [x] 确定计算账户 `naiss2026-4-1521-gpu`、分区 `gpu`，登录节点 `licheng@arrhenius1.hpc.arrhenius.naiss.se`
- [x] 在 GPU 节点上用 `manifold_steering.def` 构建镜像 `manifold_steering_20261106.sif`（4.7 GB）
- [x] main 与 train 两个环境都能用上 GPU；torch 2.14.0+cu130；`torch.compile` 正常
- [x] `safetytooling.apis` 导入正常
- [x] 4 卡 DDP 冒烟测试通过（作业 4931852，节点 n154）
- [x] `~/.bashrc` 整理为统一的 Manifold Steering 配置块；`HF_HOME` 恢复为 `hf_cache`，复用已有缓存
- [x] GitHub SSH 认证通过，代码已推送（`main` 与 `origin/main` 同步）
- [x] HF 登录（用户 AsuraLmax，Read token 存在 `$HF_HOME`）
- [x] 下载 `smoke`（Qwen2.5-0.5B-Instruct、GLP 1B 多层）、`qwen`（Qwen2.5-1.5B/7B-Instruct）、`data`（训练数据与全部评测集）三组
- [x] 添加 third_party 子模块（weight-steering、generative_latent_prior、persona_vectors）
- [x] 单卡冒烟测试全部通过（节点 n454，日志 `$PROJ/logs/smoke_gpu1_20261106-2308.log`）：vLLM 生成（需 `VLLM_USE_FLASHINFER_SAMPLER=0`）、HF 前向 hook（零向量逐字一致，4 倍范数随机向量使输出崩坏）、GLP 多层采样
- [x] **G0 完成（2026-11-06）**
- [x] `manifold_guard/layers.py`、`stats.py` 及单元测试（`tests/`，在 CPU 上几秒跑完）（2026-11-06）
- [x] Llama-3.2-1B / Llama-3.1-8B 许可已通过，下载 `llama` 组（2026-11-06）：Llama-3.2-1B-Instruct 2.4 GB、Llama-3.1-8B-Instruct 15 GB、`glp-llama8b-d6` 13 GB
- [ ] 阶段 0 MVP

### 路径与账号

| 项目 | 值 |
| --- | --- |
| 登录 | `ssh licheng@arrhenius1.hpc.arrhenius.naiss.se`（Mac 上的别名 `arrhenius`） |
| Slurm 账户 / 分区 | `naiss2026-4-1521-gpu` / `gpu` |
| `$PROJ`（与 `$BASE` 相同） | `/nobackup/proj/disk/naiss2025-22-1730/personal/licheng` |
| `$REPO` 代码仓库 | `$PROJ/Manifold_Steering_LLM`（所有代码都放这里） |
| GitHub 远端 | `git@github.com:MaxCHENGLONG/Manifold_Steering_LLM.git` |
| 当前镜像 | `$PROJ/containers/manifold_steering_current.sif` → `manifold_steering_20261106.sif` |
| 镜像定义与锁文件 | `$REPO/containers/manifold_steering.def`、`$REPO/containers/lock-20261106/` |
| HF 缓存 `$HF_HOME` | `$PROJ/hf_cache`（模型在 `hub/` 下；HF token 也存在这里） |
| 模型下载方式 | GPU 节点的交互会话里，用容器 main 环境运行 `msm python tools/download_assets.py --set ...` |
| Apptainer 缓存 / 临时目录 | `$PROJ/apptainer_cache`、`$PROJ/apptainer_tmp` |
| `TORCH_HOME` | `$PROJ/torch_cache` |
| Slurm 日志 | `$PROJ/logs` |
| GitHub SSH 密钥 | `~/.ssh/id_ed25519_github`，在 `~/.ssh/config` 里指定 |
| 备用镜像 | `$PROJ/containers/pytorch_25.11.sif`（NGC PyTorch 25.11，8.5 GB） |

### 镜像与硬件实测

- 镜像基于 `python:3.12-bookworm`，有两个环境：`/opt/venvs/main`（默认；vLLM、safetytooling、GLP 依赖、调试工具）和 `/opt/venvs/train`（trl、peft）。
- 两个环境的 torch 都是 2.14.0+cu130；NCCL 2.30.7。
- 每个节点有 4 块 NVIDIA GH200 120GB；驱动 580.196.07，最高支持 CUDA 13.0。
- 卡间两两 NV6（6 条 NVLink）全互联；1 GiB all_reduce 的 busbw 为 328 GB/s。
- GPU i 就近的 CPU 核是 72i 到 72i+71（NUMA 节点 i）；正式训练脚本用 `os.sched_setaffinity` 做绑定。

### 日常命令

```bash
# 1 卡调试会话
interactive -A naiss2026-4-1521-gpu -p gpu --gpus=1 -t 04:00:00
# 交互式进入容器（默认 main；切 train：source /opt/venvs/train/bin/activate）
apptainer shell --nv -B $PROJ $SIF
# 直接运行：main 环境 / train 环境
msm python 脚本.py
mst python 脚本.py
mst torchrun --standalone --nproc_per_node=4 train.py
# 4 卡批处理作业
cd $REPO && sbatch slurm/ddp_smoke.sbatch
# 查看作业与日志
squeue -u $USER
tail -F $PROJ/logs/<作业名>_<作业号>.out
# 提交代码
cd $REPO && git add . && git commit -m "..." && git push
```

### `~/.bashrc` 配置块

```bash
# >>> Manifold Steering >>>
export PROJ=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
export BASE=$PROJ
export REPO=$PROJ/Manifold_Steering_LLM
export APPTAINER_CACHEDIR=$PROJ/apptainer_cache
export APPTAINER_TMPDIR=$PROJ/apptainer_tmp
export HF_HOME=$PROJ/hf_cache
export TORCH_HOME=$PROJ/torch_cache
export VLLM_USE_FLASHINFER_SAMPLER=0      # 镜像里没有 nvcc，禁用需要现场编译的 FlashInfer 采样器
SIF=$PROJ/containers/manifold_steering_current.sif
msm() { apptainer exec --nv -B $PROJ $SIF "$@"; }
mst() { apptainer exec --nv -B $PROJ $SIF /opt/venvs/train/bin/"$1" "${@:2}"; }
# <<< Manifold Steering <<<
```

### 踩过的坑

重建镜像或排错时先查这张表。

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 登录节点上跑不了镜像，或构建出的镜像架构不对 | 登录节点是 x86_64，GPU 节点是 aarch64 | 只在 GPU 节点上构建和运行镜像 |
| `%test` 报 `Illegal option` | `%test` 不支持 `-c /bin/bash` | 用普通 `%test`，写 sh 语法 |
| `MS_VENV=train` 不生效 | `%environment` 读不到宿主机上临时设置的变量 | train 环境用完整路径 `/opt/venvs/train/bin/...`，即 `mst` 函数 |
| `safetytooling` 导入失败 | 缺 `google.generativeai`；新版 together 删掉了 `ServiceUnavailableError` | 不带依赖安装 google-generativeai 0.8.4、google-ai-generativelanguage 0.6.15、together 1.5.4（已写入 def） |
| axolotl 装不上 | 各版本钉死 torch 2.6/2.8 或旧版 datasets | 放弃 axolotl，LoRA 训练改用 trl + peft |
| vLLM 启动失败：`Could not find nvcc` | 默认的 FlashInfer top-k/top-p 采样器要现场编译，镜像里没有 CUDA 编译器 | 设 `VLLM_USE_FLASHINFER_SAMPLER=0`，改用 PyTorch 原生采样（已写入 `.bashrc`、冒烟脚本和 def） |
| 容器里 `which` 报 `Illegal option --` | 宿主机把 `which` 定义成了带参数的 shell 函数 | 容器里用 `command -v` |
| 整节点作业排队很久 | `--exclusive` 要等整台机器完全空闲 | 冒烟测试去掉 `--exclusive`；正式训练保留 |
| `$PROJ` 指向不对 | 原 `.bashrc` 里已有 `PROJ` 的定义 | `PROJ` 指个人目录，代码仓库用 `REPO` |
| `ssh -T git@github.com` 被拒 | 没有 `~/.ssh/config`，非默认名字的密钥不会被使用 | 写 config，指定 `id_ed25519_github` |
| scp 报 `dest open ... Failure` | 目标目录不存在 | 先 `ssh arrhenius "mkdir -p 目录"` 再上传 |

### 下一步

1. 阶段 0 MVP（Qwen2.5-1.5B-Instruct）：`manifold_guard/layers.py`、`stats.py` 已完成；接下来写逐层统计脚本，在自然语料上累计每层均值与协方差。
2. 高斯与 PCA 打分器、阈值校准；用 weight-steering 的数据算 sycophancy 的激活 steering 向量与权重 steering 向量。
3. 扫描 k，teacher-forcing 打分，画出"每层离流形分数 / 行为指标 / 能力指标"随 k 的曲线，判定 G1。

## 代码仓库结构

顶层仓库是 `Manifold_Steering_LLM`，三个上游仓库作为 git submodule 放在 `third_party/` 下，自己的代码全部放在 `manifold_guard` 包里。上游只改两处入口：加参数、接 hook，其余当库调用。目前三个 submodule 指向上游原仓库；开始改上游代码之前，先 fork 到自己的 GitHub，再把 submodule 地址换成 fork。

截至 2026-11-06，仓库里已有 `containers/`、`docs/`、`slurm/`、`smoke/`、`tools/` 和 `third_party/`；`manifold_guard/`、`scripts/`、`configs/`、`tests/` 从阶段 0 开始逐步建立。

```
Manifold_Steering_LLM/
  third_party/
    weight-steering/          # LoRA 训练、task_vectors、vLLM 评测、Claude judge
    generative_latent_prior/  # GLP 权重加载、sample_on_manifold
    persona_vectors/          # 为 Qwen 生成激活 steering 向量（.pt）
  containers/                 # 镜像定义文件与锁文件（已有）
  slurm/                      # sbatch 模板（已有：ddp_smoke.sbatch）
  smoke/                      # 冒烟测试（已有）
  tools/                      # 模型与数据集下载脚本（已有）
  docs/                       # 本计划（已有）
  manifold_guard/             # 新包；镜像只读，通过 PYTHONPATH 导入
    layers.py                 # 定位 model.layers、读写 hidden state（兼容 tuple 输出）
    stats.py                  # 流式均值/协方差，保存与加载
    scorers/                  # base.py, gauss.py, pca.py, glp.py
    correctors.py             # 投影/修正算子
    guard.py                  # ManifoldGuard 上下文管理器
    calibrate.py              # 逐层分位数阈值
    trajectory/               # M3：data.py, model.py, train.py
    reg.py                    # M4：训练时正则 ManifoldPenalty
  scripts/                    # 每个实验一个入口脚本
  configs/                    # omegaconf yaml：模型、层、打分器、阈值
  tests/
  results/                    # gitignore
```

| 文件 | 作用 | 基于 / 改自 | 阶段 | 估计行数 |
| --- | --- | --- | --- | --- |
| `manifold_guard/layers.py` | 层定位、hidden state 取出与写回 | `ActivationSteerer._locate_layer` | 0 | 60 |
| `manifold_guard/stats.py` | 逐层流式均值、协方差（float64 累加，Chan 合并） | 新写 | 0 | 120 |
| `manifold_guard/scorers/*.py` | 统一 `score(h, layer)` 接口：Mahalanobis、PCA 残差、GLP 损失 | GLP 的 `Normalizer`、`fm_prepare` | 0–2 | 250 |
| `manifold_guard/correctors.py` | PCA 投影、高斯收缩、GLP SDEdit、score 小步修正 | GLP 的 `postprocess_on_manifold_wrapper` | 2 | 150 |
| `manifold_guard/guard.py` | 全层 forward hook、门控、触发日志 | `ActivationSteererMultiple` | 2 | 200 |
| `manifold_guard/calibrate.py` | 自然激活上的分数分位数 → 阈值 τ_l | 新写 | 0 | 80 |
| `manifold_guard/trajectory/` | 跨层条件先验 p(h_l \| h_{l-1}) 的数据、模型、训练 | GLP 的去噪器结构 | 2 后半 | 400 |
| `manifold_guard/reg.py` | 训练时离流形惩罚 `ManifoldPenalty` | 新写 | 2 后半 | 80 |
| `scripts/collect_layer_stats.py` | 在语料上累计统计 | 新写 | 0 | 120 |
| `scripts/compute_diffmean_vectors.py` | 从 D⁺/D⁻ 算激活 steering 向量（persona_vectors 格式） | 新写 | 0 | 100 |
| `scripts/validate_scorers.py` | 打分器 AUROC 验证 | 新写 | 0–1 | 120 |
| `scripts/sweep_weight_steering.py` | 循环 k：调 `task_vectors.py` → vLLM 生成 → judge | `inference_and_eval.py` | 0–1 | 150 |
| `scripts/sweep_activation_steering.py` | 循环系数，可选挂 guard | `steering_inference.py` | 0–1 | 120 |
| `scripts/score_teacher_forced.py` | 把生成文本喂回模型，逐层打分 | 新写 | 0–1 | 150 |
| `scripts/eval_identity.py` | 干净文本 CE loss 增量与触发率 | 新写 | 2 | 100 |
| `scripts/train_with_manifold_reg.py` | peft + Trainer，loss = CE + λ · 惩罚 | 新写，超参取自 axolotl 配置 | 2 后半 | 150 |
| `scripts/analyze_h1_h2.py` | 汇总、相关性、Pareto 图 | 新写 | 1 | 200 |

上游需要改的两处：

- `weight-steering/steering_inference.py`：加 `--guard_config`；在 `sample_steering` 里写成 `with select_steerer(...), build_guard(...)`。guard 必须在 steerer 之后注册，因为同一层的 forward hook 按注册顺序执行。
- `weight-steering/inference_and_eval.py`：第 28–58 行把 steering 向量路径硬编码在 `/workspace/...`，改成从配置读取，并把 guard 参数透传给子进程。

## GH200 环境搭建

用一个主环境跑全部代码：新版 torch（≥ 2.11，PyPI 上已有 CUDA aarch64 wheel）加最新 vLLM，不装 flash-attn 和 bitsandbytes。GLP 只当库导入，不沿用它锁死的 torch 2.7 / vllm 0.9.2。预计 1–3 天。在 Arrhenius 上，第 2–3 步改为在构建容器镜像时完成，见下一节。

1. 确认硬件与驱动：`uname -m` 输出 `aarch64`，`nvidia-smi` 显示 GH200，驱动支持 CUDA 12.8 以上。HF 缓存目录 `HF_HOME` 放在本地 NVMe 上。
2. 建主环境：`uv venv -p 3.11`，然后装 `torch>=2.11`、`vllm`、`transformers`、`datasets`、`peft`、`accelerate`、`trl`、`diffusers`、`einops`、`omegaconf`、`safetensors`、`scikit-learn`、`pytest`、`wandb`。（实际镜像用 Python 3.12，trl 和 peft 放在单独的 train 环境里，见"镜像与硬件实测"。）
3. 装上游代码：clone `safety-research/safety-tooling` 并 `pip install -e`（weight-steering 依赖它）；GLP 用 `pip install -e third_party/generative_latent_prior --no-deps`，缺什么补什么。
4. 设置密钥：`HF_TOKEN`（Llama 是 gated 模型）、`ANTHROPIC_API_KEY`（weight-steering 的 judge）、`OPENAI_API_KEY`（persona_vectors 生成向量时的过滤与打分）、可选 `WANDB_API_KEY`。
5. 去掉硬编码路径：把 weight-steering 里所有 `/workspace/...` 换成配置项。
6. 跑五个冒烟测试，全部通过才进入阶段 0：
    1. bf16 矩阵乘与 SDPA 注意力能在 GPU 上跑。
    2. vLLM 能加载 Qwen2.5-1.5B-Instruct 并生成。
    3. `steering_inference.py` 能在 HF 上挂 `ActivationSteerer` 生成。
    4. `load_glp("generative-latent-prior/glp-llama1b-d12-multi")` 能加载，并在第 7 层采样 64 条激活。
    5. LoRA 训练（trl + peft）能在 1 块卡上跑 10 步，并在 4 块卡上以 DDP 跑通。

实际做法（2026-11-06）：G0 没有逐条照搬上面五项，而是按下面四个测试验收：`smoke/gpu1_vllm.py`（vLLM 加载 Qwen2.5-0.5B-Instruct 生成）、`smoke/gpu2_hf_hook.py`（bf16 的 HF 前向 hook，代替第 3 项里的 `steering_inference.py`）、`smoke/gpu3_glp.py`（GLP 在第 0、7、15 层各采样 64 条）、`smoke/ddp_smoke.py`（4 卡上用随机初始化的小 Qwen2 跑 5 步 LoRA DDP，代替第 5 项）。`steering_inference.py` 留到改上游入口时再测。

axolotl 在这个环境里装不上（见"踩过的坑"），LoRA 训练改用 trl + peft。超参仍从 weight-steering 的 axolotl 配置里读取（以 `qwen-2.5-1.5b (GCD)` 和 `qwen-2.5-7b` 目录为准），但下面几项要换成对应的写法：

| axolotl 配置里的原值 | 在 trl + peft 里的做法 | 原因 |
| --- | --- | --- |
| `load_in_8bit: true` | 不量化，以 bf16 加载 | 96GB 显存不需要量化 |
| `optimizer: adamw_bnb_8bit` | `optim="adamw_torch_fused"` | 避开 ARM 上的 bitsandbytes |
| `flash_attention: true` | `attn_implementation="sdpa"` | aarch64 没有 flash-attn 的上游 wheel |
| `dataset_prepared_path`、`output_dir` 指向 `/workspace/...` | 从配置读取 | 去掉硬编码路径 |
| `wandb_entity` | 换成自己的 | — |

改成 bf16 之后，训练数值会和论文略有不同。所以每个 LoRA 都按论文的做法，在 2–3 个学习率里按验证 loss 重新选一次。

如果主环境装不顺，退路是 NGC 的 arm64 PyTorch 容器，在容器里再装 vLLM 和 trl。

## 在 Arrhenius 上用容器开发与运行

所有 GPU 工作都在 Arrhenius 的 GH200 节点上、在自建的 aarch64 Apptainer 镜像里完成。代码、数据和结果留在项目目录，通过绑定挂载进容器，不打进镜像；镜像只装依赖，版本锁死、只读。

```mermaid
flowchart LR
    L["登录节点（x86_64）<br/>编辑代码、git<br/>sbatch / interactive 提交作业<br/>运行 judge（Claude / OpenAI API）"]
    G["GPU 节点（aarch64，每节点 4×GH200）<br/>构建并测试 SIF 镜像<br/>开发调试（1 卡）：apptainer shell --nv<br/>训练用整节点 4 卡；扫描与评测用 1 卡作业数组<br/>节点本地 SSD：合并权重等临时文件"]
    S[("项目存储 $PROJ（无备份）<br/>containers/ · Manifold_Steering_LLM/ · hf_cache/ · logs/")]
    L -- 提交作业 --> G
    L -- 编辑代码 --> S
    G -- 绑定挂载读写 --> S
```

登录节点只做轻量工作并提交作业；镜像构建、开发调试和所有实验都在 GPU 节点上进行，两者读写同一个项目目录。

### 系统事实与对计划的影响

| 系统事实 | 对计划的影响 | 来源 |
| --- | --- | --- |
| GPU 分区 382 个节点，每节点 4 块 GH200；每块 96 GB HBM + 128 GB LPDDR | k 扫描改用作业数组，每个任务 1 块 GPU；CPU 侧内存是 128 GB，7B 权重合并仍然够用 | [Arrhenius 入门课](https://hpc.pages.naiss.se/training/ai-intro/lessons/basics/) |
| GPU 节点是 aarch64，CPU 节点是 x86_64 | 镜像必须在 GPU 节点上构建和测试；登录节点很可能也是 x86_64，不能在上面跑这个镜像 | [AFA-Benchmark #97](https://github.com/Linusaronsson/AFA-Benchmark/issues/97) |
| 容器只能从项目目录、/scratch、/tmp 运行；缓存留在 $HOME 时直接拉取 docker:// 会失败 | SIF 全部放 `containers/`；Apptainer 缓存和临时目录指向你给的两个路径 | [NAISS Apptainer 集群设置](https://hpc.pages.naiss.se/training/basic_singularity_apptainer/cluster-setups/) |
| 每个 GPU 节点有 1.92 TB 本地 SSD | 合并后的 7B 临时权重、vLLM 与 Triton 编译缓存放节点本地，作业结束即丢弃 | [SUPR 2026 Medium](https://supr.naiss.se/round/2026medium/) |
| Arrhenius Disk 没有备份 | 代码以 git 远端为准；结果汇总定期同步到别处 | [SUPR 2026 Medium](https://supr.naiss.se/round/2026medium/) |
| Slurm 账户要带 `-gpu` 后缀，并指定 `gpu` 分区 | 所有 GPU 作业用 `naiss2026-4-1521-gpu` 和 `-p gpu` | [NAISS Slurm 课](https://hpc.pages.naiss.se/training/NAISS_Slurm/slurm/) |

存储路径属于项目 naiss2025-22-1730，计算账户是 naiss2026-4-1521。开工前用 `projinfo` 和配额命令确认两者都有效、空间和文件数配额够用。

### 目录布局

以 BASE = `/nobackup/proj/disk/naiss2025-22-1730/personal/licheng`（即 `$PROJ`）为根。下表是 2026-11-06 的实际情况。

| 目录 | 用途 | 对应环境变量 | 状态 |
| --- | --- | --- | --- |
| `containers/` | SIF 镜像，按"名称_日期"命名（如 `manifold_steering_20261106.sif`，附 `.sha256`），从不覆盖；`manifold_steering_current.sif` 指向当前在用的镜像；定义文件与锁文件进 git | — | 已有 |
| `apptainer_cache/` | Apptainer 拉取与构建缓存 | APPTAINER_CACHEDIR | 已有 |
| `apptainer_tmp/` | 构建临时目录；构建太慢时改用节点本地 /tmp | APPTAINER_TMPDIR | 已有 |
| `hf_cache/` | HF token 与模型、数据集缓存（模型在 `hub/` 下） | HF_HOME | 已有 |
| `Manifold_Steering_LLM/` | 代码仓库与三个 submodule | REPO | 已有 |
| `logs/` | Slurm 标准输出与错误输出、冒烟测试日志 | — | 已有 |
| `torch_cache/` | torch 的下载缓存 | TORCH_HOME | 尚未创建，首次使用时自动创建 |
| `cache/` | pip、Triton、TorchInductor 的持久缓存 | PIP_CACHE_DIR、TRITON_CACHE_DIR、TORCHINDUCTOR_CACHE_DIR | 待建 |
| `stats/`、`results/`、`ckpt/` | 逐层统计、实验结果、LoRA 检查点 | — | 待建 |

原计划里的 `hf_home/`（与 `hf_cache/` 分开存 token）和 `envs/`（镜像之上的薄包目录）已经建好，但没有使用，目前都是空目录；`code/manifold-steer/` 改成了 `Manifold_Steering_LLM/`。

API key（Anthropic、OpenAI、HF、wandb）放在 `$HOME` 下权限为 600 的文件里，由启动脚本读入环境变量。不放项目目录（同项目成员可能可读），不进镜像，不写进日志。

### 镜像设计

实际采用的方案（2026-11-06）：一个镜像，基于 `python:3.12-bookworm`，用 uv 建 main 和 train 两个 venv，两者的 torch 版本相同；锁文件是两个 venv 的 `pip freeze` 结果，存在 `containers/lock-20261106/`；在 GPU 节点的交互会话里直接 `apptainer build`，`%post` 自动在 fakeroot 下运行，不需要外部构建机；代码不进镜像，脚本通过 `sys.path` 或 PYTHONPATH 导入 `third_party/` 下的仓库。下面是最初的设计考虑。

- **一个还是两个镜像。** 先尝试一个镜像。如果 vLLM 锁定的 torch 版本和其他包冲突，就拆成 `ms-core`（torch、transformers、peft、accelerate、trl、diffusers、GLP 与 safety-tooling 的依赖、测试工具）和 `ms-vllm`（vLLM 及其锁定的 torch）。拆开不影响流水线，因为 weight-steering 本来就通过子进程和文件衔接生成与评测。
- **基础镜像。** 首选 NVIDIA CUDA 的 arm64 运行时镜像，再用 uv 按锁文件安装 PyPI wheel，可复现性最好。备选是站点在用的 NGC PyTorch arm64 镜像，先用它验证 `--nv`、绑定挂载和 GPU 可见性。
- **版本锁。** pyproject 与 uv 锁文件进 git。每个结果目录记录镜像名、git 哈希、Slurm 作业号和完整配置。
- **代码不进镜像。** SIF 是只读的，不能做 editable 安装，所以用 PYTHONPATH 指向 `code/` 下的各个仓库和 `envs/`。改代码不需要重建镜像，只有依赖变化才重建。
- **在哪里构建。** 在 GPU 节点的交互会话里构建。先确认 Arrhenius 是否允许 `--fakeroot` 构建；不允许的话，在一台 arm64 机器上用 Docker 构建 linux/arm64 镜像，推到镜像仓库，再在 Arrhenius 上拉成 SIF。
- **构建参数。** 按 NAISS 的建议限制 mksquashfs 的进程数与内存（8 个进程、4G），避免在共享节点上占满资源。

### 开发与运行循环

1. **登录节点**（SSH 或 ThinLinc）：编辑代码、git、提交作业。不在登录节点上跑 GPU 代码或 aarch64 镜像。模型和数据集在 GPU 节点的交互会话里用 `msm python tools/download_assets.py --set ...` 下载到 `hf_cache/`。
2. **交互开发**：用 `interactive` 申请 1 块 GPU、2–4 小时，进入 `apptainer shell --nv` 并绑定 BASE。每次先跑单元测试和冒烟测试，再调试；需要 notebook 时用 SSH 隧道连 Jupyter。
3. **批处理**：每类作业一个 sbatch 模板（构建镜像、逐层统计、LoRA 训练、扫描、打分、guard 生成）。统一由一个启动包装脚本设置环境变量、绑定路径、记录元数据。
4. **并行扫描**：k 和系数扫描用 Slurm 作业数组，每个数组任务 1 块 GPU、处理一个取值。
5. **judge 与 GPU 作业解耦**：GPU 作业只生成和打分（`--skip_judge_eval`）；judge 单独作为一个轻量步骤运行，不让 GPU 空等 batch API。
6. **离线运行**：批处理作业设 `HF_HUB_OFFLINE=1`，只读预下载的缓存；wandb 用离线模式，事后在登录节点同步。已确认 GPU 节点可以出网：镜像构建时的 pip 安装和模型下载都在 GPU 节点上完成。

每个 sbatch 模板只属于下面三种资源配置之一：调试用 1 块卡，训练用整节点 4 块卡，扫描与评测用 1 卡作业数组。

| 配置 | 用途 | 申请资源 | 要点 |
| --- | --- | --- | --- |
| 开发调试 | 交互调试、单元测试与冒烟测试、小规模试跑 | 1 块 GH200，`interactive`，2–4 小时 | 所有代码先在这里跑通，再提交批处理 |
| 训练 | LoRA（τ⁺/τ⁻、GCD、M4）、M3 的 flow 先验、可选的小 GLP | 1 个整节点（即 4 块 GH200）；固定节点数为 1 并要满 4 块卡，避免被拆到两个节点 | `torchrun` 或 accelerate 做单节点 DDP；每个进程绑定到同一块超级芯片的 CPU 核与 LPDDR |
| 扫描与评测 | k 与系数扫描、teacher-forcing 打分、guard 生成、逐层统计 | 作业数组，每个任务 1 块 GH200 | 各任务互相独立，不需要卡间通信 |

多卡训练时全局 batch 保持不变：每卡 batch × 卡数 × 梯度累积，等于论文或单卡调试时的全局 batch。权重 steering 的 LoRA 为 8，GCD 微调为 16；学习率不变。这样 4 卡训练出的 τ⁺/τ⁻ 与单卡调试结果可以直接比较。

### 对原计划的调整

- **G0** 改为：镜像在 GPU 节点上构建完成，并在容器内通过 5 个冒烟测试、全部单元测试和 4 卡 DDP 训练冒烟测试。含镜像迭代，预计 2–4 天。2026-11-06 通过；实际验收内容是 3 项单卡冒烟测试（vLLM 生成、HF 前向 hook、GLP 多层采样）和 4 卡 LoRA DDP 冒烟测试。单元测试测的是 `manifold_guard`，随阶段 0 的代码一起写。
- **GPU 时间预算**：实验矩阵合计约 15 个 GPU 日，约 360 GPU 小时；开发与重跑按翻倍估算，约 700 GPU 小时。训练作业占整节点，每墙钟小时计 4 GPU 小时，总量基本不变，只是完成得更快。先用 `projinfo` 查项目的月度配额。
- **日历时间**：有了每节点 4 块 GPU 和作业数组，阶段 1 和阶段 3 的瓶颈从 GPU 变成 judge 等待和人力。

### 上手清单（G0 之前）

2026-11-06 G0 通过时的完成情况：

- [x] 确认登录节点是 x86_64、GPU 节点是 aarch64；GPU 节点可以出网。
- [ ] 用 `projinfo` 和配额命令确认账户与配额、项目目录的空间与文件数配额。
- [x] 在 `~/.bashrc` 里设置 Apptainer、HF 和其他缓存目录的环境变量（见"`~/.bashrc` 配置块"）；`cache/`、`stats/`、`results/`、`ckpt/` 还没建，见"目录布局"。
- [x] 验证 `--nv`、绑定挂载和 GPU 可见性：已用自建镜像的冒烟测试验证。
- [x] 确认能用 fakeroot 构建：在 GPU 节点上 `apptainer build` 时 `%post` 自动在 fakeroot 下运行，不需要外部构建机。
- [x] 写定义文件和锁文件，构建第一个镜像（一个镜像、两个 venv，没有拆成两个镜像）。
- [x] 预下载 Qwen2.5-1.5B/7B-Instruct、Llama-3.2-1B、Llama-3.1-8B、GLP 权重和评测数据集（实际在 GPU 节点上用 `msm` 下载）。
- [x] 容器内跑通冒烟测试，并在 4 卡节点上用 torchrun 跑通 5 步 LoRA 训练，通过 G0（单元测试随 `manifold_guard` 一起写）。
- [ ] 写好扫描用的 sbatch 模板、作业数组模板和元数据记录，提交第一个作业：Qwen2.5-1.5B 的逐层统计。

## 模块 1：逐层激活统计

这个模块定义"自然流形"：在自然输入上逐层累计均值、协方差和特征分解，不存全部激活，只留一个小的随机样本用来校准阈值。它是所有打分器的基础，阶段 0 第一周完成。

**数据。** 一半是 FineWeb 文本，一半是对话 prompt（如 UltraChat），回答由被测模型自己用 vLLM 生成，即 on-policy。prompt token 和 response token 分开统计，因为 steering 默认只作用在 response 位置。拟合集和校准集按文档切开，互不重叠。

**必须避开的三个坑：**

- 去掉每个序列的首 token：attention sink 位置的范数比其他位置大几个数量级，会把协方差带偏。
- 用 attention mask 去掉 padding token。
- 用 forward hook 挂在 `model.layers[i]` 上取输出，不用 `output_hidden_states`：HF 返回的最后一项已经过 final norm，和 steerer 挂的位置不一致。层号约定：本包的层号 l 就是 `model.layers[l]` 的输出，和 `ActivationSteerer.layer_idx` 一致；persona 向量的第 0 行是 embedding 层，所以 `vector[l]` 对应本包的第 l-1 层。

**接口（已实现）：**

```python
# manifold_guard/layers.py
def locate_layer(model, layer_idx): ...       # 直接调用 ActivationSteerer._locate_layer，与 steerer 挂在同一个模块上
class LayerTap:
    """上下文管理器：在指定层挂 forward hook，缓存输出（兼容 tuple）。"""
    def __init__(self, model, layers="all", positions="all", detach=True): ...
    outputs: dict[int, torch.Tensor]          # layer -> (B, T, d)；positions="last" 时为 (B, 1, d)
def token_mask(attention_mask, drop_first=True): ...               # 去掉 padding 与首 token
def role_masks(attention_mask, prompt_lens, drop_first=True): ...  # {"prompt": ..., "response": ...}

# manifold_guard/stats.py
class StreamingMoments:
    def __init__(self, d, device="cuda", norm_sample=100_000, chunk_rows=16_384, seed=0): ...
    def update(self, x: torch.Tensor) -> None: ...      # x: (N, d)，已去掉 pad 与首 token
    def merge(self, other: "StreamingMoments") -> None: ...
    def finalize(self) -> dict: ...  # n, mean, cov, eigvals（从大到小）, eigvecs, norm 分位数
    def state_dict(self) -> dict: ...                   # 与 from_state_dict 配合，作业数组各任务的结果事后合并

class Reservoir:
    """每层保留固定数量的随机激活（float16），用于校准与 kNN。"""
    def __init__(self, k, d, device="cuda", dtype=torch.float16, seed=0): ...
```

`LayerTap` 的 `positions` 目前只支持 `all` 和 `last`：阶段 0 只需要这两种，prompt 和 response 用 `role_masks` 来分。

**数值做法。** 每个 chunk 先算 chunk 内均值和中心化散度矩阵，再用 Chan 合并公式并入总量，全程 float64（原计划是块内 float32、合并时 float64）。GH200 上 FP64 矩阵乘和不开 TF32 的 FP32 一样快，全程 float64 还不受全局 TF32 设置影响。最后在 GPU 上用 float64 的 `torch.linalg.eigh` 做特征分解。实测 d=3584 时，每 16384 个 token 的一次更新约 11 ms，2000 万 token × 28 层的统计部分约 7 分钟（不含模型前向）；GPU 与 CPU 结果的相对误差为 2e-16。

**保留样本跨层对齐。** `Reservoir` 用 bottom-k 抽样：每行抽一个随机键，始终保留键最小的 k 行。seed 和分块方式相同时，各层保留的是同一批 token，可以逐行对上，M3 需要这一点（见 `test_reservoir_same_seed_selects_same_positions`）。作业数组里各任务的结果要合并时，各任务要用不同的 seed。

**输出。** 每层一个文件 `stats/<model>/<role>/layer_XX.pt`，含 n、mean、cov、eigvals、eigvecs，再加每层 5 万条保留样本。7B 的统计量以 float32 存储，两个角色合计约 6 GB；保留样本每个角色约 10 GB。

**命令行：**

```
python scripts/collect_layer_stats.py --model Qwen/Qwen2.5-7B-Instruct \
  --mix fineweb:0.5,chat_onpolicy:0.5 --n_tokens 2e7 --layers all \
  --roles prompt,response --reservoir 50000 --out stats/qwen7b
```

**验收：** 单元测试里协方差与 numpy 结果的相对误差 < 1e-6；分块合并与一次算完结果一致；7B 跑 2000 万 token 在 GH200 上不超过 2 小时；每层的特征值谱已画图存档。

## 模块 2：离流形打分器

所有打分器实现同一个接口，分数越大越离流形；四种打分器从便宜到昂贵排列，门控用便宜的，复核用 GLP。分数在每层按自然样本的中位数和四分位距标准化，这样不同层、不同打分器可以直接比较。

```python
# manifold_guard/scorers/base.py
class Scorer(Protocol):
    name: str
    def load(self, stats_dir: str) -> None: ...
    def score(self, h: torch.Tensor, layer: int) -> torch.Tensor: ...  # (N, d) -> (N,)

# manifold_guard/calibrate.py
def calibrate(scorer, reservoir_dir, layers, qs=(0.9, 0.99, 0.999)) -> dict:
    """在校准集上算每层分数分位数和标准化参数，存成 json。"""

def trajectory_score(per_layer: dict[int, torch.Tensor], agg="max") -> torch.Tensor:
    """把各层标准化分数聚合成一条轨迹的分数（max 或 mean）。"""
```

| 打分器 | 定义 | 每 token 每层开销 | 适用模型 | 用途 |
| --- | --- | --- | --- | --- |
| 范数比 | ‖h‖ 除以自然样本的中位范数 | 可忽略 | 全部 | 基线，检查是不是只测到了范数 |
| 高斯（Mahalanobis） | 用收缩协方差 (1−λ)Σ + λ·(tr Σ / d)·I 白化后的平方距离，除以 d | 一次 d×d 矩阵乘 | 全部 | 门控主力 |
| PCA 残差 | 去掉前 k 个主成分后的剩余能量，k 取解释 99% 方差的维数 | 一次 d×k 矩阵乘 | 全部 | 门控，与投影配套 |
| GLP 损失 | 固定 u（如 0.3、0.5、0.7）、4 次噪声平均的 flow-matching 损失 | 约等于一次完整 LLM 前向，或更多 | 仅 Llama（1B 全层，8B 第 15 层） | 离线复核、M0 |

GLP 分数的调用方式：用 `glp.normalizer.normalize(x[:, None, :], layer_idx=l)` 归一化，再调 `glp(latents=..., u=..., layer_idx=l, loss_kwargs={"reduction": "none"}).loss`，在后两维上取平均。

**打分器自身的验证（`scripts/validate_scorers.py`）。** 正样本是校准集的自然激活。负样本分三类：

- 自然激活加上随机方向的扰动，范数与 steering 向量在各个 k 下的范数相同；
- 真实的激活 steering 结果，覆盖各个系数；
- 把激活的维度随机打乱（保持范数和边缘分布）。

每层、每个打分器报告 AUROC。**验收：** 在能力已下降 10% 以上的系数下，至少一个便宜打分器的 AUROC ≥ 0.9；范数比打分器明显更差，否则说明其他打分器只测到了范数。

## 模块 3：ManifoldGuard（M0/M1/M2）

ManifoldGuard 是一个和 `ActivationSteerer` 同形的上下文管理器：在选定层挂 forward hook，逐 token 打分，超过阈值才修正，其余位置原样返回。M0、M1、M2 是同一个类的三组配置，不是三套代码。

```python
# manifold_guard/guard.py
class ManifoldGuard:
    def __init__(self, model, scorer, corrector, thresholds: dict[int, float],
                 layers="all", positions="response",
                 preserve: torch.Tensor | None = None,   # 要保留的 steering 方向
                 log_path: str | None = None): ...
    def __enter__(self): ...   # 在 model.layers[l] 上注册 hook，必须晚于 steerer
    def __exit__(self, *exc): ...  # 移除 hook，写出日志
    stats: dict                # 每层触发率、平均修正幅度 ‖Δh‖/‖h‖

def build_guard(model, cfg) -> "ManifoldGuard | nullcontext": ...

# manifold_guard/correctors.py
def pca_project(h, layer, alpha=1.0): ...        # 去掉 alpha 比例的主子空间外残差
def gauss_shrink(h, layer, tau, preserve=None): ...  # 在白化空间里把偏离缩回到 tau，保留 preserve 方向的分量
def score_step(h, layer, eta): ...               # h += eta * ∇log p(h)，高斯时为 -eta·Σ⁻¹(h-μ)
def glp_sdedit(h, layer, u=0.5, steps=20): ...   # 包装 GLP 的 postprocess_on_manifold_wrapper
```

| 模式 | 作用层 | 门控 | 默认修正算子 | 预期结果 |
| --- | --- | --- | --- | --- |
| M0 | 单层（steering 层；Llama-3.1-8B 为第 15 层） | 无 | `glp_sdedit`，u=0.5、20 步 | 复现 GLP 的流畅度提升，作为基线 |
| M1 | 全部层 | 无 | `pca_project` 或 `glp_sdedit` | 量化误差逐层累积的代价，预期过不了恒等性检查 |
| M2 | 全部层 | 分数超过该层 99% 分位数才修正 | `gauss_shrink`（保留 steering 方向）；触发时可换成 `glp_sdedit` | 主方法 |

**hook 内的处理顺序：**

1. 取出这一层的输出；`positions="response"` 时只处理最后一个 token，和 steerer 的语义一致。
2. 整批一次矩阵乘算出标准化分数，不写 Python 循环。
3. 用布尔掩码挑出超过 τ_l 的行，只对这些行调用修正算子，所以开销和触发率成正比。
4. 把修正后的行写回输出 tuple，记录触发数和 ‖Δh‖/‖h‖。

修正后的状态会进入后续层和 KV cache，这正是我们想要的。对激活 steering，默认只在 steering 层及之后挂 guard；对权重 steering，所有层都可能漂移，所以全部挂上。

配置示例：

```yaml
guard:
  mode: M2                  # none | M0 | M1 | M2
  scorer: gauss             # norm | gauss | pca | glp
  corrector: gauss_shrink   # pca_project | gauss_shrink | score_step | glp_sdedit
  layers: from_steer_layer  # all | from_steer_layer | [15]
  gate_quantile: 0.99
  preserve_steering_direction: true
  positions: response
  stats_dir: stats/qwen7b/response
```

**验收：** 阈值设成无穷大时 guard 的输出与不挂 guard 逐位相同；不加 steering 时 M2 满足恒等性判据；M0 在 Llama-3.1-8B-Instruct 的 persona 向量 steering 上复现出流畅度提升的方向。

## 模块 4：steering 扫描与 teacher-forcing 打分（H1/H2）

这个模块回答 H1 和 H2，而且生成时不挂任何 hook：先用 vLLM 或 HF 正常生成，再把"prompt + 生成的回答"喂回同一个被 steering 的模型，用 `LayerTap` 逐层打分。这样权重 steering 可以全程用 vLLM，速度快很多。

**激活 steering 向量的来源。** 论文对两种方法用同一份 D⁺/D⁻。所以新增 `scripts/compute_diffmean_vectors.py`：直接在 weight-steering 训练 LoRA 用的那几个 HF 数据集上，用 `LayerTap` 算 response token 的均值差。输出格式和 persona_vectors 的 `.pt` 一致（第 0 行是 embedding 层）。这样 Qwen 线不需要 OpenAI key；如果手头有 persona_vectors 生成的向量，就拿来做一致性核对。

**一次扫描的步骤（每个 k 或系数）：**

1. 权重 steering：调 `task_vectors.py`，把 θ_pre + k·(θ⁺ − θ⁻) 存成临时模型。7B 每个约 15 GB，用完即删。激活 steering：跳过这一步。
2. 生成：权重 steering 用 `vllm_inference.py`；激活 steering 用 `steering_inference.py`。
3. 评测：调 `inference_and_eval.py --skip_model_inference`，走原来的 Claude judge。
4. 打分：`score_teacher_forced.py` 在固定的 500 条子集上，把文本喂回模型并逐层打分。激活 steering 打分时，steerer 要用和生成时一致的位置掩码：最后一个 prompt token 加全部 response token。
5. 写结果，删临时权重。

```python
# scripts/score_teacher_forced.py
def score_teacher_forced(model, tok, records, scorers, layers, steerer=None,
                         roles=("response",), batch_size=16) -> "pd.DataFrame":
    """返回列：example_id, token_idx, layer, scorer, score, z, over_tau"""
```

每个配置还要打两个对照：

- **自然基线**：k=0 的原模型在它自己的回答上的分数，决定 0 点在哪里。
- **文本是否自然**：把 steering 后生成的文本喂给原模型打分。这个分数高，说明文本本身就不自然；它低而主测量高，说明不自然的是 steering 后模型的内部计算。

**初始扫描网格（沿用论文的范围，阶段 0 后再加密）：**

| 行为 | 模型 | 权重 steering 的 k | 激活 steering 系数（层） |
| --- | --- | --- | --- |
| sycophancy | Qwen2.5-1.5B-Instruct | 待阶段 0 定 | 待阶段 0 定（论文 GCD 实验选第 17 层） |
| sycophancy | Qwen2.5-7B-Instruct | −28 到 +15 | −5 到 +5（第 12 层），外加全层变体 |
| evil | Qwen2.5-7B-Instruct | 1 到 30 | 0.5 到 6（第 17 层），外加全层变体 |

**结果格式。** `results/<line>/<behavior>/<method>/<k>/` 下放 `responses.json`、`judge.json`、`scores.parquet`。

**分析（`scripts/analyze_h1_h2.py`）：**

- H1：每种方法在 k 方向上，算轨迹分数与能力指标的 Spearman 相关。两条曲线各做一次分段线性拟合，比较拐点的位置。
- H2：用单调插值把两种方法对齐到相同的行为水平（例如 evil 率 0.3、0.5、0.7），在这些点上比较轨迹分数，对 prompt 做 bootstrap 得到置信区间。

**验收：** 在 Qwen2.5-1.5B 上用 5 个 k 跑通整条流水线，耗时 3 小时以内；在 7B sycophancy 上复现论文的定性结论，即降低 sycophancy 时权重 steering 优于激活 steering。

## 模块 5：跨层轨迹先验 M3

M3 不问"h_l 像不像第 l 层的激活"，而问"从 h_{l−1} 走到 h_l 这一步自不自然"。一个状态可能单看在第 l 层的边缘分布里，却不是 h_{l−1} 的合理后继；逐层的边缘先验（包括多层 GLP）看不出这种情况。先做闭式的线性高斯版，再做非线性的条件 flow 版，后者只有明显更好才保留。

```python
# manifold_guard/trajectory/model.py
class TransitionPrior(nn.Module):
    def score(self, h_prev, h_cur, layer: int) -> torch.Tensor: ...    # (N,)
    def correct(self, h_prev, h_cur, layer: int, **kw) -> torch.Tensor: ...

class LinearGaussianTransition(TransitionPrior):
    """h_l ≈ A_l h_{l-1} + b_l + ε，ε ~ N(0, S_l)；分数是残差在 S_l 下的马氏距离。"""

class FlowTransition(TransitionPrior):
    """条件 flow matching：以 h_{l-1} 和层号为条件，对 Δ_l = h_l - h_{l-1} 建模。"""
```

**线性高斯版（约 1 周）。** 把 `StreamingMoments` 扩展成相邻两层的联合矩：多累计一个交叉协方差 E[h_l h_{l−1}ᵀ]。由联合矩直接解出 A_l、b_l、S_l，不需要训练。7B 每对相邻层的交叉协方差约 100 MB（float64），27 对放在 Grace 的 CPU 内存里没有压力。

**条件 flow 版（约 3–4 周）：**

1. 先在 Llama-3.2-1B 上做（16 层、d=2048），因为它有现成的多层 GLP，可以直接比较"边缘先验"和"条件先验"。
2. 去噪器沿用 GLP 的 MLP 结构，用 FiLM 注入 h_{l−1} 的条件，层号沿用 GLP 的正弦编码。
3. 数据边产边训：同一进程里交替做 LLM 前向（产出 1–2 亿对 (h_{l−1}, h_l)）和去噪器训练，这是 1 卡调试时的做法。正式训练在 4 卡节点上直接用 GLP 的 producer-consumer 流水线：1 块卡跑 LLM 前向产出激活对，其余 3 块卡用 DDP 训练去噪器。
4. 模型规模先定 0.5B 参数左右，4 卡节点上预计半天到 1 天（单卡约 1–2 天）。

**接入 guard。** 在第 l−1 层的 hook 里缓存修正后的输出，第 l 层的 hook 读取它作为条件。两者在同一次前向内，所以不需要额外的前向。

**验收与止损：**

- 在模块 2 的负样本上，条件分数的 AUROC 至少比边缘分数高 0.03，尤其是小范数 steering 的情形。
- 在 M2 的 Pareto 曲线上，用 M3 打分器替换边缘打分器后，前沿要外移。
- FlowTransition 没有比 LinearGaussianTransition 的 AUROC 高 0.03 以上，就只保留线性版。

## 模块 6：训练时流形正则 M4

M4 把约束从推理时挪到训练时：任务微调时加一项惩罚，让微调后模型的激活不越出原模型的自然分布，看能否直接减少论文里的"行为漂移"。用自己写的小训练脚本（peft + Trainer），不用 axolotl；LoRA 超参与论文一致。

```python
# manifold_guard/reg.py
class ManifoldPenalty(nn.Module):
    """用原模型的高斯统计（冻结）。惩罚 = mean softplus(z_l(h) - τ_l)，只算 response token。"""
    def __init__(self, stats_dir, layers, tau: dict[int, float], scorer="gauss"): ...
    def forward(self, hidden: dict[int, torch.Tensor], mask: torch.Tensor) -> torch.Tensor: ...

# scripts/train_with_manifold_reg.py
# peft + transformers.Trainer，重写 compute_loss：loss = ce + λ · penalty
# LoRA：r=32, alpha=16, 全部 linear；GCD 用 lr 2e-4、1 epoch（同论文表 1）
```

**实现要点：**

- 用不 detach 的 `LayerTap` 取 hidden state，每隔一层取一次，以节省显存。
- 马氏距离用预先算好的白化矩阵实现，整条路径可微。
- λ 扫 0.01、0.1、1 三档；τ_l 用模块 2 校准的 99% 分位数。

**测试平台与对比组：**

| 组 | 做法 | 来源 |
| --- | --- | --- |
| A | GCD 微调（query-only + correct-solution） | 论文 4.1.2 |
| B | Joint：A 的数据加上 D⁻ | 论文 4.1.2 |
| C | A 之后做事后权重 steering | 论文 4.1.2 |
| D | A 加上 M4，λ 三档 | 本计划 |
| E | D 之后再做事后权重 steering | 本计划 |

主平台是 Qwen2.5-1.5B 的 GCD-sycophancy。第二平台是 Llama-2-7b-chat 的 GSM8K 微调导致拒答退化（论文 4.3），时间允许再做。

**预期与验收：** D 保留 A 的 GCD 提升的 95% 以上，同时在留出的对话 prompt 上降低离流形分数。sycophancy 不一定会降：GCD 数据教出的谄媚行为可能本来就在原模型的自然分布里。如果真是这样，就把它作为负结果写进论文，说明"留在流形上"约束的是连贯性而不是对齐。

## 评估与实验矩阵

评估完全沿用论文的数据和 judge，只新增两项：恒等性和流畅度。开发阶段每个配置只跑 500 条子集，最终选出的点才跑全量，以控制 judge 的 API 开销和等待时间。

| 评测 | 数据（HF 数据集） | 指标 | 判定方式 |
| --- | --- | --- | --- |
| 事实类 sycophancy | `cfierro/sycophancy_eval_answer`（TruthfulQA + TriviaQA，4 种提示） | 基线准确率、非谄媚率（正确性） | Claude judge（论文 D.1） |
| GCD sycophancy | `cfierro/gcd` | 无提示正确率、非谄媚率（正确性 / 不同意） | Claude judge（论文 D.2–D.3） |
| evil（World Affecting） | `cfierro/ethical_world_affecting`（零样本）、`_cot`、`_cot-tags`、`_cot-same-mmlu` | 有效且 evil 的比例、CoT 一致性 | 规则解析 + Claude judge（论文 D.4） |
| 能力保持 | `cfierro/tiny_mmlu_chat`（TinyMMLU）；代码中另有 `cfierro/mmlu_chat` | MMLU 准确率、有效回答率 | 规则解析 |
| 分布内 evil | `cfierro/persona-vectors-eval-questions` | evil 分数、连贯性 | Claude judge（persona_vectors 提示，论文 G.4） |
| 拒答（可选） | `cfierro/DirectHarm4-chat`、`cfierro/GSM-Danger_chat`、`cfierro/gsm8k_instr_chat` | 安全率、ASR、GSM8K 正确率 | Claude judge（论文 D.5） |
| 流畅度 | 上述开放式回答 | 0–100 连贯性分数 | LLM judge（persona_vectors 的 coherence 提示） |
| 恒等性 | FineWeb 留出 100 万 token + 模型自己的对话回答 | CE loss 增量、门控触发率 | 直接计算 |

**实验矩阵（GPU 时间为 GH200 上的粗估，不含 judge 等待）：**

| 编号 | 内容 | 模型 | 方法 × 配置 | GPU 时间 |
| --- | --- | --- | --- | --- |
| E0 | 恒等性检查 | 4 个模型 | 不加 steering × {M1, M2-高斯, M2-PCA}；Llama 另加 GLP 版 | 0.5 天 |
| E1 | H1/H2：sycophancy | Qwen2.5-7B | 权重 / 激活 / 激活全层 / bias-only × k 网格，全部做 teacher-forcing 打分 | 1 天 |
| E2 | H1/H2：evil | Qwen2.5-7B | 同 E1，外加 TinyMMLU | 1 天 |
| E3 | guard 用于激活 steering | Qwen2.5-7B；Llama-3.1-8B | Qwen：{无, M2-高斯, M2-PCA}；Llama：persona 向量 × {无, M0, M2-GLP 门控} | 2 天 |
| E4 | guard 用于权重 steering | Qwen2.5-7B | 权重 steering 后的模型 × {无, M2}，HF 生成 | 1–2 天 |
| E5 | 全层 GLP | Llama-3.2-1B-Instruct | 激活与权重 steering × {M1-GLP, M2-GLP}，用多层 GLP | 2 天 |
| E6 | 跨层先验 | Llama-3.2-1B；Qwen2.5-7B（仅线性版） | M2 × {边缘打分器, M3 线性, M3 flow} | 3–5 天（含训练） |
| E7 | 训练时正则 | Qwen2.5-1.5B | GCD 的 A–E 五组，λ 三档 | 1 天 |
| E8 | 鲁棒性（可选） | Llama-3.1-8B | persona 越狱、针对 guard 的自适应攻击 | 2 天 |

E5 需要 Llama-3.2-1B 自己的 D⁺/D⁻：按论文流程用 system prompt 让模型生成，再把论文里 GPT-4.1-mini 的过滤换成 Claude judge。采样类评测跑 3 个随机种子；evil 评测按论文用贪心解码。

## 里程碑与时间表

按单人全职估算共 16 周；第 5 周末的 G2 是整个计划唯一的去留判断点，之后的方法工作是否全力投入取决于它。

| 工作线 | 周次 | 结束时的关卡 |
| --- | --- | --- |
| 环境 + 阶段 0：MVP | 第 1–2 周 | G1 |
| 阶段 1：H1/H2 诊断 | 第 3–5 周 | **G2：H1/H2 是否成立（关键决策）** |
| 阶段 2a：guard M0–M2 | 第 6–8 周 | G3：M2 过恒等性 |
| 阶段 2b：M3 跨层先验 | 第 9–12 周 | G4：M3 止损判定 |
| M4：训练时正则 | 第 11–12 周 | — |
| 阶段 3/4：全量评估 | 第 13–15 周 | — |
| 整理与写作 | 第 16 周 | — |

M4 与 M3 的后半段重叠：M3 的 flow 模型训练时，人力转去写 M4，1.5B 的 LoRA 训练可以和它共用 GPU。

| 关卡 | 时间 | 通过条件 | 不通过时 |
| --- | --- | --- | --- |
| G0 | 第 1 周末 | 镜像在 GPU 节点上构建完成，容器内 5 个冒烟测试与单元测试全部通过，4 卡 DDP 训练能跑通 | 换 NGC arm64 容器，或为 GLP 单独建环境 |
| G1 | 第 2 周末 | Qwen2.5-1.5B 上离流形分数的拐点与能力崩坏的拐点对得上 | 先升级打分器（加 GLP 复核或换范数归一化后的分数），再进阶段 1 |
| G2 | 第 5 周末 | H1 的 \|ρ\| ≥ 0.7，H2 的方向成立 | 记为负结果；方法部分缩减为 M0 与 M2 的对照实验 |
| G3 | 第 8 周末 | M2 通过恒等性检查，并在至少一个行为上把 Pareto 前沿外推 | 调门控分位数与修正步长；仍不过就把 guard 定位为诊断工具 |
| G4 | 第 12 周末 | M3 的 AUROC 比边缘打分器高 ≥ 0.03 | 按止损规则只保留线性版 |

## 测试清单与风险应对

单元测试全部用一个随机初始化的小 Llama 配置在 CPU 上跑，几秒内完成；每次改动 `manifold_guard` 都必须先过这一套。

- [x] `test_stats`：`StreamingMoments` 与 numpy 的均值和协方差一致；分块合并与一次算完一致。
- [x] `test_layertap`：hook 取到的输出等于 HF `hidden_states[l+1]`（最后一层除外）；能处理 tuple 输出。
- [ ] `test_guard_identity`：阈值为无穷大时，贪心生成结果与不挂 guard 逐 token 相同。
- [ ] `test_hook_order`：同一层先注册 steerer、后注册 guard，guard 看到的是 steering 之后的值。
- [ ] `test_positions`：`response` 模式在 prefill 时只改最后一个 prompt token，在 decode 时只改新 token。
- [ ] `test_scorers`：在合成高斯数据上，马氏距离服从卡方分布；落在主子空间内的点 PCA 残差为 0。
- [ ] `test_glp_wrapper`：u 趋近 0 时 `glp_sdedit` 的输出约等于输入。
- [ ] `test_task_vectors`：k=0 时合并模型的 logits 与原模型相同。
- [ ] `test_diffmean`：有 persona_vectors 的 `.pt` 时，自算向量与它的余弦相似度 ≥ 0.99。

| 风险 | 早期信号 | 应对 |
| --- | --- | --- |
| aarch64 上依赖装不上（safetytooling、GLP 的版本锁） | 环境冒烟测试失败 | GLP 只导入核心模块；换 NGC arm64 容器；必要时为 GLP 单独建环境 |
| 便宜打分器只测到了范数 | 范数比打分器的 AUROC 和其他打分器差不多 | 先对 h 做范数归一化再打分；复核改用 GLP 分数 |
| 恒等性检查不过 | 自然输入下触发率 > 1% 或 CE 增量 > 1% | 提高门控分位数；只在 steering 层之后挂 guard；减小修正步长 |
| HF 生成太慢 | E3/E4 单个配置超过半天 | 只用 500 条子集；静态 KV cache 加 `torch.compile`；只修正 response 位置 |
| H1/H2 不成立 | 阶段 1 的相关系数很弱 | 作为负结果记录；换 GLP 分数重测；再决定方法部分是否继续投入 |
| judge 的费用与等待 | batch 积压超过一天 | 开发阶段用子集；先用便宜模型预筛，只对关键点用 Claude Sonnet |
| 磁盘被临时权重占满 | 7B 合并模型每个约 15 GB | 用完即删；激活只保留统计量和小样本，不整体落盘 |
| GLP 只有 Llama 权重 | Qwen 线做不了 GLP 打分 | 接受这一限制；或在 Qwen 上训练一个 1–2 亿条激活的小 GLP，约 1 天 |
| M3 的 flow 版不如线性版 | AUROC 提升 < 0.03 | 按止损规则只保留线性版 |
