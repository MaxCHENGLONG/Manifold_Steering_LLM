"""打分器的公共接口：分数越大越离流形。统计量来自 collect_layer_stats.py 写出的 layer_XX.pt。"""
from pathlib import Path
from typing import Protocol

import torch


class Scorer(Protocol):
    name: str

    def load(self, stats_dir, layers=None, device="cuda") -> "Scorer": ...

    def score(self, h: torch.Tensor, layer: int) -> torch.Tensor: ...  # (N, d) -> (N,)


def stats_layers(stats_dir):
    """统计目录里有哪些层（包内层号，即 model.layers 的下标）。"""
    return sorted(int(p.stem.split("_")[1]) for p in Path(stats_dir).glob("layer_*.pt"))


class LayerwiseScorer:
    """每层独立打分。子类实现 _prepare（由该层统计量算出打分用的张量）和 _score。"""
    name = ""

    def load(self, stats_dir, layers=None, device="cuda"):
        self.device = device
        layers = stats_layers(stats_dir) if layers is None else list(layers)
        if not layers:
            raise FileNotFoundError(f"{stats_dir} 里没有 layer_*.pt")
        self.params = {}
        for l in layers:
            s = torch.load(Path(stats_dir) / f"layer_{l:02d}.pt", map_location="cpu")
            self.params[l] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in self._prepare(s).items()}
        return self

    @property
    def layers(self):
        return sorted(self.params)

    def score(self, h, layer):
        return self._score(h.to(self.device, torch.float32), self.params[layer])

    def _prepare(self, stats):
        raise NotImplementedError

    def _score(self, h, p):
        raise NotImplementedError
