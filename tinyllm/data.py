from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ShardWriter:
    def __init__(self, output_dir: str | Path, shard_tokens: int, with_mask: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.with_mask = with_mask
        self.tokens: list[int] = []
        self.mask: list[int] = []
        self.shards: list[dict[str, int | str]] = []

    def add(self, tokens: list[int], mask: list[int] | None = None) -> None:
        if self.with_mask and (mask is None or len(tokens) != len(mask)):
            raise ValueError("SFT shards require a mask matching the token count")
        self.tokens.extend(tokens)
        if mask is not None:
            self.mask.extend(mask)
        while len(self.tokens) >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def close(self) -> dict[str, object]:
        if self.tokens:
            self._flush(len(self.tokens))
        manifest = {"tokens": sum(int(s["tokens"]) for s in self.shards), "shards": self.shards}
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def _flush(self, count: int) -> None:
        index = len(self.shards)
        token_name = f"tokens-{index:05d}.bin"
        np.asarray(self.tokens[:count], dtype=np.uint16).tofile(self.output_dir / token_name)
        entry: dict[str, int | str] = {"tokens": count, "file": token_name}
        del self.tokens[:count]
        if self.with_mask:
            mask_name = f"mask-{index:05d}.bin"
            np.asarray(self.mask[:count], dtype=np.uint8).tofile(self.output_dir / mask_name)
            entry["mask"] = mask_name
            del self.mask[:count]
        self.shards.append(entry)


def deterministic_split(key: str, validation_percent: int = 1) -> str:
    import hashlib

    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little") % 100
    return "validation" if bucket < validation_percent else "train"


class PackedDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        seed: int = 1_337,
        use_mask: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.seed = seed
        self.use_mask = use_mask
        manifest = json.loads((self.data_dir / "manifest.json").read_text())
        self.shards = manifest["shards"]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        seed = self.seed + (worker.id if worker else 0)
        rng = random.Random(seed)
        shards = list(self.shards)
        while True:
            rng.shuffle(shards)
            for shard in shards:
                tokens = np.memmap(self.data_dir / shard["file"], dtype=np.uint16, mode="r")
                mask = None
                if self.use_mask:
                    mask = np.memmap(self.data_dir / shard["mask"], dtype=np.uint8, mode="r")
                offsets = list(range(0, len(tokens) - self.seq_len + 1, self.seq_len))
                rng.shuffle(offsets)
                for offset in offsets:
                    ids = torch.from_numpy(np.array(tokens[offset : offset + self.seq_len], dtype=np.int64))
                    if mask is None:
                        labels = ids.clone()
                    else:
                        selected = np.array(mask[offset : offset + self.seq_len], dtype=bool)
                        if not selected.any():
                            continue
                        labels = ids.clone()
                        labels[~torch.from_numpy(selected)] = -100
                    yield ids, labels
