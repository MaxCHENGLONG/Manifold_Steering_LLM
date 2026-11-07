import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from manifold_guard.calibrate import calibrate, standardize, trajectory_score
from manifold_guard.corpus import Sequence
from manifold_guard.layers import LayerTap
from manifold_guard.scorers import GaussScorer, NormScorer
from manifold_guard.stats import StreamingMoments
from scripts.generate_steered import sycophancy_subset
from scripts.score_teacher_forced import QS, score_records

D, N_LAYERS = 32, 4
LAYERS = list(range(N_LAYERS))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=D, intermediate_size=64, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def setup(model, tmp_path_factory):
    """用模型在随机序列上的激活做逐层统计（每层的保留样本逐行对齐），再校准两个打分器。"""
    root = tmp_path_factory.mktemp("stats")
    ids = torch.randint(0, 128, (64, 32), generator=torch.Generator().manual_seed(1))
    with torch.no_grad(), LayerTap(model, LAYERS) as tap:
        model(input_ids=ids)
    for l in LAYERS:
        h = tap.outputs[l].reshape(-1, D).double()
        m = StreamingMoments(D, device="cpu")
        m.update(h[:1500])
        s = {k: v.float() if torch.is_tensor(v) else v for k, v in m.finalize().items()}
        torch.save({"role": "response", "layer": l, **s}, root / f"layer_{l:02d}.pt")
        torch.save({"role": "response", "layer": l, "samples": h[1500:].half(), "seen": len(h) - 1500},
                   root / f"reservoir_{l:02d}.pt")
    scorers = [NormScorer().load(root, device="cpu"), GaussScorer().load(root, device="cpu")]
    return scorers, {s.name: calibrate(s, root) for s in scorers}


@pytest.fixture(scope="module")
def seqs():
    g = torch.Generator().manual_seed(2)
    out = []
    for k, (n, p) in enumerate([(12, 4), (20, 9), (7, 3)]):
        out.append(Sequence("chat", str(k), torch.randint(0, 128, (n,), generator=g).tolist(), p))
    return out


def _rows(res):
    return {(e, l, s): i for i, (e, l, s) in enumerate(zip(res["example"], res["layer"], res["scorer"]))}


def test_matches_unbatched_reference(model, setup, seqs):
    """一起 padding 成一个 batch 打分，与逐条不 padding、直接按定义算出的统计一致。"""
    scorers, calibs = setup
    res = score_records(model, seqs, scorers, calibs, LAYERS, pad_id=0, batch_tokens=1000)
    idx = _rows(res)
    assert len(idx) == len(seqs) * (N_LAYERS + 1) * len(scorers)
    for s in seqs:
        with torch.no_grad(), LayerTap(model, LAYERS) as tap:
            model(input_ids=torch.tensor([s.input_ids]))
        for sc in scorers:
            c = calibs[sc.name]
            z = {l: standardize(sc.score(tap.outputs[l][0, s.prompt_len:], l), c, l) for l in LAYERS}
            ref = {l: (z[l], c["layers"][l]["z_quantiles"]) for l in LAYERS}
            ref[-1] = (trajectory_score(z, "max"), c["trajectory"]["max"]["quantiles"])
            for l, (zl, thr) in ref.items():
                i = idx[(int(s.doc_id), l, sc.name)]
                assert res["n_tokens"][i] == len(s.input_ids) - s.prompt_len
                assert res["z_mean"][i] == pytest.approx(zl.mean().item(), rel=1e-4, abs=1e-4)
                assert res["z_max"][i] == pytest.approx(zl.max().item(), rel=1e-4, abs=1e-4)
                for q in QS:
                    assert res[f"frac_q{q[2:]}"][i] == pytest.approx((zl > thr[q]).double().mean().item())


def test_sycophancy_subset_keeps_groups():
    rows = [{"base": {"question": f"q{g}"}} for g in range(10) for _ in range(4)]
    a = sycophancy_subset(rows, 3, seed=0)
    assert a == sycophancy_subset(rows, 3, seed=0) and len(a) == 12
    for i in range(0, 12, 4):
        assert a[i] % 4 == 0 and a[i: i + 4] == list(range(a[i], a[i] + 4))
    rows[5]["base"]["question"] = "other"
    with pytest.raises(ValueError):
        sycophancy_subset(rows, 3, seed=0)
