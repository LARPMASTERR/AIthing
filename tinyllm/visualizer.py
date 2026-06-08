from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tinyllm.model import TinyLM
from tinyllm.tokenizer import TinyTokenizer


def layout_cache_key(metadata: dict[str, str | int], tokenizer_path: str | Path) -> str:
    tokenizer_hash = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()[:12]
    value = f"{metadata['phase']}-{metadata['step']}-{tokenizer_hash}"
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def embedding_layout(
    model: TinyLM,
    tokenizer: TinyTokenizer,
    metadata: dict[str, str | int],
    tokenizer_path: str | Path,
    cache_dir: str | Path = "artifacts/visualizer",
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"layout-{layout_cache_key(metadata, tokenizer_path)}.json"
    if path.exists():
        return path

    embeddings = model.embedding.weight.detach().float().cpu()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    positions = centered @ eigenvectors[:, -3:]
    scale = torch.quantile(positions.abs(), 0.99, dim=0).clamp_min(1e-6)
    positions = (positions / scale * 55).clamp(-70, 70)

    payload = {
        "checkpoint": metadata,
        "count": len(tokenizer),
        "positions": positions.round(decimals=4).tolist(),
        "labels": [tokenizer.token_text(token_id) for token_id in range(len(tokenizer))],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(path)
    return path
