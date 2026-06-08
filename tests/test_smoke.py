from __future__ import annotations

from tinyllm.checkpoint import best_checkpoint, load_model
from tinyllm.config import DataConfig, ModelConfig, PathsConfig, ProjectConfig, TrainConfig
from tinyllm.data import ShardWriter
from tinyllm.model import generate
from tinyllm.tokenizer import TinyTokenizer
from tinyllm.train import train_phase


def test_tiny_end_to_end_training(tmp_path):
    tokenizer = TinyTokenizer.train(
        iter(["A tiny model learns from a tiny corpus."] * 100),
        tmp_path / "tokenizer.json",
        vocab_size=512,
    )
    text_tokens = tokenizer.encode("A tiny model learns from a tiny corpus. " * 100)
    pretrain_dir = tmp_path / "pretrain"
    writer = ShardWriter(pretrain_dir, 256)
    writer.add(text_tokens)
    writer.close()

    sft_dir = tmp_path / "sft"
    writer = ShardWriter(sft_dir, 256, with_mask=True)
    writer.add(text_tokens, [1] * len(text_tokens))
    writer.close()

    quick_train = TrainConfig(
        batch_size=1,
        grad_accum_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=0,
        max_steps=1,
        max_seconds=30,
        save_every=1,
    )
    config = ProjectConfig(
        model=ModelConfig(
            vocab_size=512,
            max_seq_len=16,
            n_layers=1,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
        ),
        data=DataConfig(shard_tokens=256),
        pretrain=quick_train,
        sft=quick_train,
        paths=PathsConfig(
            tokenizer=str(tmp_path / "tokenizer.json"),
            pretrain_data=str(pretrain_dir),
            sft_data=str(sft_dir),
            checkpoints=str(tmp_path / "checkpoints"),
            system_prompt=str(tmp_path / "system.txt"),
        ),
    )
    train_phase(config, "pretrain")
    train_phase(config, "sft")
    model = load_model(best_checkpoint(config), device="cpu")
    output = generate(model, tokenizer.encode("A tiny"), max_new_tokens=2, temperature=0)
    assert len(output) == 2


def test_zero_second_limit_still_saves_checkpoint(tmp_path):
    tokens = list(range(64))
    data_dir = tmp_path / "pretrain"
    writer = ShardWriter(data_dir, 64)
    writer.add(tokens)
    writer.close()
    config = ProjectConfig(
        model=ModelConfig(
            vocab_size=64,
            max_seq_len=8,
            n_layers=1,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            d_ff=32,
        ),
        data=DataConfig(shard_tokens=64),
        pretrain=TrainConfig(max_seconds=0),
        paths=PathsConfig(
            pretrain_data=str(data_dir),
            checkpoints=str(tmp_path / "checkpoints"),
        ),
    )
    checkpoint = train_phase(config, "pretrain")
    assert checkpoint.exists()
