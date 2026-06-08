from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi

from tinyllm.chat import Message, encode_sft
from tinyllm.config import ProjectConfig
from tinyllm.data import ShardWriter
from tinyllm.tokenizer import TinyTokenizer

SOURCES = {
    "fineweb_edu": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        "license": "ODC-By-1.0; also subject to Common Crawl terms",
    },
    "smol_smoltalk": {
        "dataset": "HuggingFaceTB/smol-smoltalk",
        "url": "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk",
        "license": "Apache-2.0",
    },
    "oasst1": {
        "dataset": "OpenAssistant/oasst1",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "license": "Apache-2.0",
    },
}


def source_manifest(names: Iterable[str]) -> list[dict[str, str]]:
    api = HfApi()
    result = []
    for name in names:
        source = dict(SOURCES[name])
        try:
            source["revision"] = api.dataset_info(source["dataset"]).sha
        except Exception as error:
            source["revision"] = f"unresolved: {error}"
        result.append(source)
    return result


def fineweb_texts(limit: int | None = None) -> Iterator[str]:
    source = SOURCES["fineweb_edu"]
    dataset = load_dataset(source["dataset"], source["config"], split="train", streaming=True)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            return
        text = row.get("text", "").strip()
        if len(text) >= 200:
            yield text


def prepare_tokenizer(config: ProjectConfig) -> TinyTokenizer:
    print(f"training tokenizer from {config.data.tokenizer_texts:,} FineWeb-Edu documents")
    texts = fineweb_texts(config.data.tokenizer_texts)
    tokenizer = TinyTokenizer.train(texts, config.paths.tokenizer, config.model.vocab_size)
    print(f"saved {len(tokenizer):,}-token tokenizer to {config.paths.tokenizer}")
    return tokenizer


def prepare_pretrain(config: ProjectConfig, tokenizer: TinyTokenizer) -> None:
    output_dir = Path(config.paths.pretrain_data)
    writer = ShardWriter(output_dir, config.data.shard_tokens)
    eot = tokenizer.token_id("<|eot|>")
    total = 0
    for text in fineweb_texts():
        tokens = tokenizer.encode(text) + [eot]
        remaining = config.data.pretrain_tokens - total
        if remaining <= 0:
            break
        tokens = tokens[:remaining]
        writer.add(tokens)
        total += len(tokens)
        if total and total % 5_000_000 < len(tokens):
            print(f"packed {total:,}/{config.data.pretrain_tokens:,} pretraining tokens")
    packed = writer.close()
    write_data_manifest(
        output_dir,
        "pretrain",
        config,
        source_manifest(["fineweb_edu"]),
        packed,
    )


def valid_messages(raw: Iterable[dict[str, str]]) -> list[Message] | None:
    messages = []
    for item in raw:
        role = item.get("role")
        content = item.get("content", "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            return None
        messages.append(Message(role, content))
    if not messages or not any(message.role == "assistant" for message in messages):
        return None
    return messages


def smoltalk_conversations(limit: int) -> Iterator[list[Message]]:
    dataset = load_dataset(SOURCES["smol_smoltalk"]["dataset"], split="train", streaming=True)
    count = 0
    for row in dataset:
        messages = valid_messages(row["messages"])
        if messages is None:
            continue
        yield messages
        count += 1
        if count >= limit:
            return


def oasst_conversations(limit: int) -> Iterator[list[Message]]:
    dataset = load_dataset(SOURCES["oasst1"]["dataset"], split="train", streaming=True)
    children: dict[str, list[dict]] = defaultdict(list)
    roots = []
    for row in dataset:
        if row.get("lang") != "en" or row.get("deleted") or row.get("review_result") is False:
            continue
        parent_id = row.get("parent_id")
        if parent_id:
            children[parent_id].append(row)
        elif row.get("role") == "prompter":
            roots.append(row)

    random.Random(1_337).shuffle(roots)
    for root in roots[:limit]:
        conversation = [Message("user", root["text"])]
        current = root
        while children.get(current["message_id"]):
            ranked = sorted(
                children[current["message_id"]],
                key=lambda child: (child.get("rank") is None, child.get("rank") or 0),
            )
            current = ranked[0]
            role = "assistant" if current["role"] == "assistant" else "user"
            conversation.append(Message(role, current["text"]))
        if any(message.role == "assistant" for message in conversation):
            yield conversation


def prepare_sft(config: ProjectConfig, tokenizer: TinyTokenizer) -> None:
    output_dir = Path(config.paths.sft_data)
    writer = ShardWriter(output_dir, config.data.shard_tokens, with_mask=True)
    smol_limit = int(config.data.sft_conversations * 0.75)
    oasst_limit = config.data.sft_conversations - smol_limit
    total_conversations = 0
    total_tokens = 0
    conversations = list(smoltalk_conversations(smol_limit))
    conversations.extend(oasst_conversations(oasst_limit))
    random.Random(config.data.seed).shuffle(conversations)
    for messages in conversations:
        tokens, mask = encode_sft(tokenizer, messages)
        if len(tokens) < 8 or not any(mask):
            continue
        writer.add(tokens + [tokenizer.token_id("<|eot|>")], mask + [1])
        total_conversations += 1
        total_tokens += len(tokens) + 1
    packed = writer.close()
    packed["conversations"] = total_conversations
    write_data_manifest(
        output_dir,
        "sft",
        config,
        source_manifest(["smol_smoltalk", "oasst1"]),
        packed,
    )
    print(f"packed {total_conversations:,} conversations and {total_tokens:,} SFT tokens")


def write_data_manifest(
    output_dir: Path,
    kind: str,
    config: ProjectConfig,
    sources: list[dict[str, str]],
    packed: dict[str, object],
) -> None:
    manifest = {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "settings": config.to_dict()["data"],
        "packed": packed,
    }
    (output_dir / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenizer and packed training data")
    parser.add_argument("stage", choices=["tokenizer", "pretrain", "sft", "all"])
    parser.add_argument("--config", default="configs/quick.json")
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)

    tokenizer_path = Path(config.paths.tokenizer)
    if args.stage in {"tokenizer", "all"}:
        tokenizer = prepare_tokenizer(config)
    elif tokenizer_path.exists():
        tokenizer = TinyTokenizer.load(tokenizer_path)
    else:
        raise SystemExit(f"missing tokenizer at {tokenizer_path}; prepare it first")

    if args.stage in {"pretrain", "all"}:
        prepare_pretrain(config, tokenizer)
    if args.stage in {"sft", "all"}:
        prepare_sft(config, tokenizer)


if __name__ == "__main__":
    main()
