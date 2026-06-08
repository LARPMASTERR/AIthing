from __future__ import annotations

import os
from pathlib import Path

import torch

from tinyllm.config import ModelConfig, ProjectConfig
from tinyllm.model import TinyLM


def checkpoint_path(config: ProjectConfig, phase: str) -> Path:
    return Path(config.paths.checkpoints) / f"{phase}-latest.pt"


def save_checkpoint(
    path: str | Path,
    model: TinyLM,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    phase: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {
        "model_config": model.config_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "phase": phase,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(
    path: str | Path,
    model: TinyLM,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


def load_model(path: str | Path, device: torch.device) -> TinyLM:
    model, _ = load_model_with_metadata(path, device)
    return model


def load_model_with_metadata(path: str | Path, device: torch.device) -> tuple[TinyLM, dict[str, str | int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyLM(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    metadata = {
        "path": str(path),
        "phase": str(payload.get("phase", "unknown")),
        "step": int(payload.get("step", 0)),
    }
    return model, metadata


def checkpoint_metadata(path: str | Path) -> dict[str, str | int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "phase": str(payload.get("phase", "unknown")),
        "step": int(payload.get("step", 0)),
    }


def best_checkpoint(config: ProjectConfig) -> Path:
    sft = checkpoint_path(config, "sft")
    pretrain = checkpoint_path(config, "pretrain")
    if sft.exists():
        return sft
    if pretrain.exists():
        return pretrain
    raise FileNotFoundError("no checkpoint found; run pretraining and SFT first")
