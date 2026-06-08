from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from tinyllm.checkpoint import load_model_with_metadata


class BrowserModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.browser_forward(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the trained TinyLM for browser inference")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/sft-latest.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("pages/model"))
    parser.add_argument("--layout", type=Path, help="Optional cached visualizer layout to publish")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model, metadata = load_model_with_metadata(args.checkpoint, torch.device("cpu"))
    model = BrowserModel(model.half()).eval()
    example = torch.ones((1, 32), dtype=torch.long)
    output_path = args.output / "tinyllm.onnx"
    torch.onnx.export(
        model,
        (example,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits", "layer_activity", "attention"],
        dynamic_axes={
            "input_ids": {1: "sequence"},
            "attention": {1: "sequence"},
        },
        opset_version=18,
        dynamo=False,
    )
    shutil.copy2(args.tokenizer, args.output / "tokenizer.json")
    config = {
        "checkpoint": {"phase": metadata["phase"], "step": metadata["step"]},
        "max_seq_len": model.model.config.max_seq_len,
        "vocab_size": model.model.config.vocab_size,
        "n_layers": model.model.config.n_layers,
        "special_tokens": {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "eom": "<|eom|>",
            "eot": "<|eot|>",
        },
        "default_system_prompt": "You are a small helpful assistant. Be concise and honest.",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.layout:
        layout = json.loads(args.layout.read_text())
        layout["checkpoint"] = config["checkpoint"]
        layout_path = args.output.parent / "data" / "layout.json"
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        layout_path.write_text(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))
    size = output_path.stat().st_size / 1024 / 1024
    if size >= 100:
        raise RuntimeError(f"{output_path} is {size:.1f} MiB; GitHub rejects files at or above 100 MiB")
    print(f"exported browser model to {output_path} ({size:.1f} MiB)")


if __name__ == "__main__":
    main()
