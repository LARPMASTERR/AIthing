from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 16_384
    max_seq_len: int = 1_024
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1_408
    rope_theta: float = 10_000.0

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")


@dataclass
class DataConfig:
    tokenizer_texts: int = 50_000
    pretrain_tokens: int = 120_000_000
    sft_conversations: int = 30_000
    shard_tokens: int = 5_000_000
    seed: int = 1_337


@dataclass
class TrainConfig:
    batch_size: int = 4
    grad_accum_steps: int = 16
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 100_000
    max_seconds: int = 5_400
    save_every: int = 250
    weight_decay: float = 0.1
    grad_clip: float = 1.0


@dataclass
class PathsConfig:
    tokenizer: str = "artifacts/tokenizer.json"
    pretrain_data: str = "data/packed/pretrain"
    sft_data: str = "data/packed/sft"
    checkpoints: str = "artifacts/checkpoints"
    system_prompt: str = "system_prompt.txt"


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pretrain: TrainConfig = field(default_factory=TrainConfig)
    sft: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            batch_size=2,
            grad_accum_steps=16,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            warmup_steps=20,
            max_steps=10_000,
            max_seconds=900,
            save_every=100,
        )
    )
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        raw = json.loads(Path(path).read_text())
        config = cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            pretrain=TrainConfig(**raw.get("pretrain", {})),
            sft=TrainConfig(**raw.get("sft", {})),
            paths=PathsConfig(**raw.get("paths", {})),
        )
        config.model.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

