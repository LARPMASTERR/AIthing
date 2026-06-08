from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinyllm.checkpoint import checkpoint_path, load_training_checkpoint, save_checkpoint
from tinyllm.config import ProjectConfig, TrainConfig
from tinyllm.data import PackedDataset
from tinyllm.model import TinyLM


def learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def make_optimizer(model: TinyLM, config: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    kwargs = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": config.weight_decay,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def train_phase(config: ProjectConfig, phase: str) -> Path:
    if phase not in {"pretrain", "sft"}:
        raise ValueError("phase must be pretrain or sft")
    train_config = config.pretrain if phase == "pretrain" else config.sft
    data_dir = config.paths.pretrain_data if phase == "pretrain" else config.paths.sft_data
    use_mask = phase == "sft"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = TinyLM(config.model).to(device)
    optimizer = make_optimizer(model, train_config, device)
    output = checkpoint_path(config, phase)
    step = 0

    if output.exists():
        step = load_training_checkpoint(output, model, optimizer)
        print(f"resumed {phase} from step {step:,}")
    elif phase == "sft":
        pretrain = checkpoint_path(config, "pretrain")
        if not pretrain.exists():
            raise FileNotFoundError("SFT requires a pretraining checkpoint")
        load_training_checkpoint(pretrain, model)
        print(f"initialized SFT from {pretrain}")

    dataset = PackedDataset(data_dir, config.model.max_seq_len, config.data.seed, use_mask)
    loader = DataLoader(dataset, batch_size=train_config.batch_size, num_workers=2, pin_memory=device.type == "cuda")
    iterator = iter(loader)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(
        f"{phase}: {model.parameter_count():,} parameters on {device}; "
        f"time limit {train_config.max_seconds / 60:.1f} minutes"
    )
    try:
        while step < train_config.max_steps and time.monotonic() - started < train_config.max_seconds:
            accumulated_loss = 0.0
            timed_out = False
            for _ in range(train_config.grad_accum_steps):
                if time.monotonic() - started >= train_config.max_seconds:
                    timed_out = True
                    break
                input_ids, labels = next(iterator)
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast:
                    _, loss = model(input_ids, labels)
                    if loss is None:
                        raise RuntimeError("training loss was not calculated")
                    loss = loss / train_config.grad_accum_steps
                loss.backward()
                accumulated_loss += float(loss.detach())

            if timed_out:
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            lr = learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or step % 10 == 0:
                elapsed = time.monotonic() - started
                print(f"step {step:,} loss {accumulated_loss:.4f} lr {lr:.2e} elapsed {elapsed:.0f}s")
            if step % train_config.save_every == 0:
                save_checkpoint(output, model, optimizer, step, phase)
                print(f"saved {output}")
    except KeyboardInterrupt:
        print("training interrupted; saving current state")
    finally:
        save_checkpoint(output, model, optimizer, step, phase)
        print(f"saved final {phase} state at step {step:,} to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyLM")
    parser.add_argument("phase", choices=["pretrain", "sft"])
    parser.add_argument("--config", default="configs/quick.json")
    args = parser.parse_args()
    train_phase(ProjectConfig.load(args.config), args.phase)


if __name__ == "__main__":
    main()
