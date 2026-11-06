"""逐层激活的流式统计：均值、协方差，以及固定容量的随机保留样本。

不存全部激活：StreamingMoments 只累计 n、均值和中心化散度矩阵 M2，Reservoir 只留一小份
均匀随机样本，用来校准阈值和做 kNN。两者都能 merge，也都能存成 state_dict，所以作业数组
里各任务的结果可以事后合并。
"""
import torch

NORM_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


class Reservoir:
    """从数据流里均匀随机地保留至多 k 行（不放回）。

    每行抽一个 U(0,1) 的随机键，始终保留键最小的 k 行（bottom-k 抽样）。批量更新不需要逐行
    循环；两个 Reservoir 合并时再取一次最小的 k 个键，结果仍是整个流上的均匀样本，前提是
    两者的 seed 不同。

    同一个 seed、同样的分块方式喂同一串 token 时，各层选中的是同一批 token 位置，
    所以不同层的保留样本可以逐行对齐（跨层先验 M3 需要这一点）。
    """

    def __init__(self, k, d, device="cuda", dtype=torch.float16, seed=0):
        self.k, self.d, self.dtype = k, d, dtype
        self.keys = torch.empty(0, dtype=torch.float64, device=device)
        self.data = torch.empty(0, d, dtype=dtype, device=device)
        self.seen = 0
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def update(self, x):
        """x: (N, d)。"""
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x 的形状应为 (N, {self.d})，实际为 {tuple(x.shape)}")
        if len(x) == 0:
            return
        keys = torch.rand(len(x), dtype=torch.float64, device=self.keys.device, generator=self.generator)
        self.seen += len(x)
        self._offer(keys, x.to(self.keys.device, self.dtype))

    def merge(self, other):
        if (other.k, other.d) != (self.k, self.d):
            raise ValueError("只能合并 k 和 d 都相同的 Reservoir")
        self.seen += other.seen
        self._offer(other.keys.to(self.keys.device), other.data.to(self.keys.device, self.dtype))

    def _offer(self, keys, x):
        free = self.k - len(self.keys)
        if free > 0:  # 还没满：直接收下
            self.keys = torch.cat([self.keys, keys[:free]])
            self.data = torch.cat([self.data, x[:free]])
            keys, x = keys[free:], x[free:]
            if len(keys) == 0:
                return
        # 已满：只有键比当前最大键小的行才可能进来；只覆写被挤掉的槽位，不整体拷贝 data
        hit = keys < self.keys.max()
        keys, x = keys[hit], x[hit]
        if len(keys) == 0:
            return
        all_keys = torch.cat([self.keys, keys])
        keep = torch.zeros(len(all_keys), dtype=torch.bool, device=all_keys.device)
        keep[torch.topk(all_keys, self.k, largest=False).indices] = True
        evicted = (~keep[: self.k]).nonzero().squeeze(1)
        incoming = keep[self.k:].nonzero().squeeze(1)
        self.keys[evicted] = keys[incoming]
        self.data[evicted] = x[incoming]

    def state_dict(self):
        return {"k": self.k, "d": self.d, "keys": self.keys.cpu(), "data": self.data.cpu(),
                "seen": self.seen, "rng": self.generator.get_state()}

    @classmethod
    def from_state_dict(cls, state, device="cuda"):
        r = cls(state["k"], state["d"], device=device, dtype=state["data"].dtype)
        r.keys = state["keys"].to(device)
        r.data = state["data"].to(device)
        r.seen = state["seen"]
        r.generator.set_state(state["rng"])
        return r


class StreamingMoments:
    """逐层流式均值与协方差。

    每块数据先算块内均值和中心化散度矩阵，再用 Chan 等人的合并公式并入总量，避免
    "先求平方和再减均值平方"带来的数值抵消。全程 float64：GH200 上 FP64 矩阵乘和不用
    TF32 的 FP32 一样快，又不受全局 TF32 设置的影响。

    另用一个 Reservoir 保留一部分 token 的范数，用来算范数分位数。
    """

    def __init__(self, d, device="cuda", norm_sample=100_000, chunk_rows=16_384, seed=0):
        self.d = d
        self.n = 0
        self.mean = torch.zeros(d, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.norms = Reservoir(norm_sample, 1, device=device, dtype=torch.float32, seed=seed)
        self.chunk_rows = chunk_rows

    def update(self, x):
        """x: (N, d)，调用方已去掉 padding 和首 token（见 layers.token_mask）。"""
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x 的形状应为 (N, {self.d})，实际为 {tuple(x.shape)}")
        for chunk in x.split(self.chunk_rows):  # 分块只为限制显存峰值，不影响结果
            chunk = chunk.to(self.mean.device, torch.float64)
            mean = chunk.mean(dim=0)
            xc = chunk - mean
            self._merge(len(chunk), mean, xc.T @ xc)
            self.norms.update(chunk.norm(dim=1, keepdim=True))

    def merge(self, other):
        if other.d != self.d:
            raise ValueError("只能合并维度相同的 StreamingMoments")
        self._merge(other.n, other.mean.to(self.mean.device), other.m2.to(self.mean.device))
        self.norms.merge(other.norms)

    def _merge(self, n_b, mean_b, m2_b):
        if n_b == 0:
            return
        n = self.n + n_b
        delta = mean_b - self.mean
        self.m2 += m2_b + torch.outer(delta, delta) * (self.n * n_b / n)
        self.mean += delta * (n_b / n)
        self.n = n

    def finalize(self, quantiles=NORM_QUANTILES):
        """返回 n、mean、cov（无偏）、eigvals（从大到小）、eigvecs（第 i 列对应 eigvals[i]）和范数分位数。"""
        if self.n < 2:
            raise ValueError("至少需要 2 个样本才能算协方差")
        cov = self.m2 / (self.n - 1)
        cov = (cov + cov.T) / 2  # 消除舍入造成的微小不对称，eigh 只读一个三角
        eigvals, eigvecs = torch.linalg.eigh(cov)
        q = torch.tensor(quantiles, dtype=torch.float32, device=cov.device)
        norm_q = torch.quantile(self.norms.data[:, 0], q)
        return {
            "n": self.n,
            "mean": self.mean.clone(),
            "cov": cov,
            "eigvals": eigvals.flip(0),
            "eigvecs": eigvecs.flip(1),
            "norm_quantiles": dict(zip(quantiles, norm_q.tolist())),
        }

    def state_dict(self):
        return {"d": self.d, "n": self.n, "mean": self.mean.cpu(), "m2": self.m2.cpu(),
                "norms": self.norms.state_dict()}

    @classmethod
    def from_state_dict(cls, state, device="cuda"):
        m = cls(state["d"], device=device, norm_sample=state["norms"]["k"])
        m.n = state["n"]
        m.mean = state["mean"].to(device)
        m.m2 = state["m2"].to(device)
        m.norms = Reservoir.from_state_dict(state["norms"], device=device)
        return m
