from __future__ import annotations

import json

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.visualizer import embedding_layout


def test_layout_is_cached_and_checkpoint_aware(tmp_path, tokenizer):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.tokenizer.save(str(tokenizer_path))
    model = TinyLM(
        ModelConfig(
            vocab_size=len(tokenizer),
            max_seq_len=16,
            n_layers=1,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
        )
    )
    metadata = {"phase": "pretrain", "step": 10, "path": "checkpoint.pt"}
    first = embedding_layout(model, tokenizer, metadata, tokenizer_path, tmp_path / "layouts")
    second = embedding_layout(model, tokenizer, metadata, tokenizer_path, tmp_path / "layouts")
    changed = embedding_layout(
        model,
        tokenizer,
        {**metadata, "step": 11},
        tokenizer_path,
        tmp_path / "layouts",
    )
    assert first == second
    assert first != changed
    layout = json.loads(first.read_text())
    assert layout["count"] == len(tokenizer)
    assert len(layout["positions"]) == len(layout["labels"]) == len(tokenizer)
    assert all(len(position) == 3 for position in layout["positions"])
