from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tinyllm.tokenizer import TinyTokenizer

VALID_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"invalid role: {self.role}")
        if not self.content.strip():
            raise ValueError("message content cannot be empty")


def normalize_messages(messages: Iterable[Message | dict[str, str]]) -> list[Message]:
    normalized = []
    for message in messages:
        normalized.append(message if isinstance(message, Message) else Message(**message))
    return normalized


def format_messages(messages: Iterable[Message | dict[str, str]], add_assistant: bool = False) -> str:
    parts = []
    for message in normalize_messages(messages):
        parts.append(f"<|{message.role}|>\n{message.content.strip()}<|eom|>\n")
    if add_assistant:
        parts.append("<|assistant|>\n")
    return "".join(parts)


def encode_sft(
    tokenizer: TinyTokenizer,
    messages: Iterable[Message | dict[str, str]],
) -> tuple[list[int], list[int]]:
    token_ids: list[int] = []
    loss_mask: list[int] = []
    for message in normalize_messages(messages):
        prefix = tokenizer.encode(f"<|{message.role}|>\n")
        content = tokenizer.encode(f"{message.content.strip()}<|eom|>\n")
        token_ids.extend(prefix)
        token_ids.extend(content)
        predict = 1 if message.role == "assistant" else 0
        loss_mask.extend([0] * len(prefix))
        loss_mask.extend([predict] * len(content))
    return token_ids, loss_mask


def build_prompt(
    tokenizer: TinyTokenizer,
    messages: Iterable[Message | dict[str, str]],
    max_tokens: int,
) -> list[int]:
    normalized = normalize_messages(messages)
    system = [message for message in normalized if message.role == "system"]
    conversation = [message for message in normalized if message.role != "system"]
    latest_user = next((m for m in reversed(conversation) if m.role == "user"), None)
    required = system + ([latest_user] if latest_user else [])
    assistant_prefix = tokenizer.encode("<|assistant|>\n")

    def parts(message: Message) -> tuple[list[int], list[int], list[int]]:
        return (
            tokenizer.encode(f"<|{message.role}|>\n"),
            tokenizer.encode(message.content.strip()),
            tokenizer.encode("<|eom|>\n"),
        )

    required_parts = [parts(message) for message in required]
    overhead = len(assistant_prefix) + sum(len(prefix) + len(suffix) for prefix, _, suffix in required_parts)
    content_budget = max(0, max_tokens - overhead)
    contents = [content for _, content, _ in required_parts]
    content_limits = [0] * len(contents)
    while content_budget and any(content_limits[i] < len(content) for i, content in enumerate(contents)):
        for index, content in enumerate(contents):
            if content_budget == 0:
                break
            if content_limits[index] < len(content):
                content_limits[index] += 1
                content_budget -= 1

    required_encoded = []
    for index, ((prefix, content, suffix), message) in enumerate(zip(required_parts, required)):
        limit = content_limits[index]
        selected_content = content[-limit:] if message is latest_user and limit else content[:limit]
        required_encoded.append(prefix + selected_content + suffix)

    history_ids = []
    for message in reversed(conversation):
        if message is latest_user:
            continue
        encoded = tokenizer.encode(format_messages([message]))
        required_length = sum(len(item) for item in required_encoded)
        if required_length + len(history_ids) + len(encoded) + len(assistant_prefix) <= max_tokens:
            history_ids = encoded + history_ids
    system_ids = [token_ids for token_ids, message in zip(required_encoded, required) if message.role == "system"]
    latest_ids = required_encoded[-1] if latest_user else []
    prompt = [token for item in system_ids for token in item]
    prompt += history_ids + latest_ids + assistant_prefix
    return prompt[-max_tokens:]
