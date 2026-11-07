"""从 HF 缓存里直接读数据集的 parquet 分片。

上游用 datasets.load_dataset 读 cfierro/... 数据集。这些仓库是用 snapshot_download 下载的，
在离线作业（HF_HUB_OFFLINE=1）里 load_dataset 找不到它们，所以这里直接读 parquet。
"""
import glob

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def load_split(repo, split):
    """返回 data/<split>-*.parquet 里的全部行（list[dict]）。"""
    root = snapshot_download(repo, repo_type="dataset")
    files = sorted(glob.glob(f"{root}/data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{repo} 里没有 {split} 分片（{root}/data/{split}-*.parquet）")
    return [row for f in files for row in pq.read_table(f).to_pylist()]
