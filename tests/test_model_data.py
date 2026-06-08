from __future__ import annotations

import torch

from tinyllm.checkpoint import load_model, save_checkpoint
from tinyllm.config import ModelConfig
from tinyllm.data import PackedDataset, ShardWriter
from tinyllm.model import TinyLM, generate, generate_traced


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=512,
        max_seq_len=32,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )


def test_model_shapes_loss_causality_and_generation():
    torch.manual_seed(7)
    model = TinyLM(tiny_model_config()).eval()
    ids = torch.randint(0, 512, (2, 12))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 12, 512)
    assert loss is not None and torch.isfinite(loss)

    changed = ids.clone()
    changed[:, 8:] = torch.randint(0, 512, changed[:, 8:].shape)
    changed_logits, _ = model(changed)
    assert torch.allclose(logits[:, :8], changed_logits[:, :8], atol=1e-5)
    assert len(generate(model, ids[0].tolist(), max_new_tokens=3, temperature=0)) == 3


def test_traced_forward_matches_logits_and_reports_signals():
    torch.manual_seed(11)
    model = TinyLM(tiny_model_config()).eval()
    ids = torch.randint(0, 512, (1, 12))
    normal, _ = model(ids)
    traced, activity, attention = model.traced_forward(ids)
    assert torch.equal(normal, traced)
    assert len(activity) == len(attention) == model.config.n_layers
    assert all(torch.isfinite(torch.tensor(activity)))
    assert all(0 <= target["position"] < ids.shape[1] for layer in attention for target in layer)

    event = next(generate_traced(model, ids[0].tolist(), max_new_tokens=1, temperature=0))
    assert 0 <= event["probability"] <= 1
    assert 0 <= event["entropy"] <= 1
    assert len(event["alternatives"]) == 5
    probabilities = [alternative["probability"] for alternative in event["alternatives"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_browser_forward_matches_latest_logits_and_reports_signals():
    torch.manual_seed(13)
    model = TinyLM(tiny_model_config()).eval()
    ids = torch.randint(0, 512, (1, 12))
    normal, _ = model(ids)
    logits, activity, attention = model.browser_forward(ids)
    assert torch.equal(normal[:, -1, :], logits)
    assert activity.shape == (model.config.n_layers,)
    assert attention.shape == (model.config.n_layers, ids.shape[1])
    assert torch.isfinite(activity).all()
    assert torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(dim=-1), torch.ones(model.config.n_layers))


def test_shards_masks_and_checkpoint(tmp_path):
    data_dir = tmp_path / "packed"
    writer = ShardWriter(data_dir, shard_tokens=32, with_mask=True)
    writer.add(list(range(48)), [0] * 8 + [1] * 40)
    manifest = writer.close()
    assert manifest["tokens"] == 48

    ids, labels = next(iter(PackedDataset(data_dir, seq_len=8, use_mask=True)))
    assert ids.shape == labels.shape == (8,)
    assert (labels != -100).any()

    model = TinyLM(tiny_model_config())
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, None, 3, "test")
    loaded = load_model(path, torch.device("cpu"))
    assert loaded.config_dict() == model.config_dict()
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), loaded.parameters()))
