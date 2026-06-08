from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from tinyllm.tokenizer import TinyTokenizer

PROMPT = "What is Linux? Name one Linux distribution."
ANSWER = (
    "Linux is an open-source operating system kernel used by many complete operating systems. "
    "Arch Linux is one Linux distribution."
)


def token_event(
    tokenizer: TinyTokenizer,
    token_id: int,
    position: int,
    context: list[int],
    generated: list[int],
    rng: random.Random,
) -> dict:
    confidence = 0.12 + rng.random() * 0.56
    alternatives = []
    for candidate in rng.sample(range(len(tokenizer)), 5):
        alternatives.append(
            {
                "token_id": candidate,
                "token_text": tokenizer.token_text(candidate),
                "probability": confidence * (0.38 - len(alternatives) * 0.055),
            }
        )
    combined = context + generated
    attention = []
    for layer in range(8):
        targets = []
        for offset in range(3):
            target_position = max(0, len(combined) - 1 - ((layer + 1) * (offset + 1)) % max(1, len(combined)))
            targets.append(
                {
                    "position": target_position,
                    "weight": 0.18 - offset * 0.035,
                    "token_id": combined[target_position],
                }
            )
        attention.append(targets)
    activity = [0.65 + math.sin(position * 0.37 + layer * 0.61) * 0.18 + layer * 0.025 for layer in range(8)]
    return {
        "type": "token",
        "token_id": token_id,
        "token_text": tokenizer.token_text(token_id),
        "text": tokenizer.decode(generated + [token_id]),
        "position": position,
        "probability": confidence,
        "entropy": 0.42 + rng.random() * 0.3,
        "alternatives": alternatives,
        "layer_activity": activity,
        "attention_targets": attention,
        "context_offset": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export safe static assets for the GitHub Pages demo")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("pages/data"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    layout = json.loads(args.layout.read_text())
    layout["checkpoint"] = {"phase": "public-demo", "step": layout.get("checkpoint", {}).get("step", 0)}
    (args.output / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))

    tokenizer = TinyTokenizer.load(args.tokenizer)
    prompt_tokens = tokenizer.encode(PROMPT)
    answer_tokens = tokenizer.encode(ANSWER)
    rng = random.Random(1_337)
    events = [
        {"type": "ready", "checkpoint": layout["checkpoint"], "sources": []},
        *[
            {
                "type": "prompt_token",
                "token_id": token_id,
                "token_text": tokenizer.token_text(token_id),
                "position": position,
            }
            for position, token_id in enumerate(prompt_tokens)
        ],
    ]
    generated = []
    for position, token_id in enumerate(answer_tokens):
        events.append(token_event(tokenizer, token_id, position, prompt_tokens, generated, rng))
        generated.append(token_id)
    events.append({"type": "done", "text": ANSWER, "token_count": len(answer_tokens)})
    payload = {"prompt": PROMPT, "answer": ANSWER, "events": events}
    (args.output / "demo-events.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"exported public demo assets to {args.output}")


if __name__ == "__main__":
    main()
