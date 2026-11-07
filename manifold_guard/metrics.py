import torch


def auroc(pos, neg):
    """负样本分数高于正样本的概率（Mann–Whitney U），并列算一半。分数越大越像负样本。

    在 GPU 上算平均秩，几十万个分数也只要几毫秒。steering 层之前的激活与基线逐位相同，结果正好 0.5。
    """
    x = torch.cat([pos.flatten(), neg.flatten()]).double()
    _, inv, counts = torch.unique(x, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    ranks = (ends - (counts.double() - 1) / 2)[inv]  # 从 1 起的平均秩
    n0, n1 = pos.numel(), neg.numel()
    u = ranks[n0:].sum() - n1 * (n1 + 1) / 2
    return (u / (n0 * n1)).item()
