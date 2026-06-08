from __future__ import annotations

from pathlib import Path

import requests
import torch

from tinyllm.chat import Message, build_prompt, normalize_messages
from tinyllm.checkpoint import best_checkpoint, load_model_with_metadata
from tinyllm.config import ProjectConfig
from tinyllm.model import TinyLM, generate, generate_traced
from tinyllm.retrieval import RetrievedPage, WikipediaRetriever, format_retrieval_context
from tinyllm.tokenizer import TinyTokenizer


class ChatEngine:
    def __init__(
        self,
        model: TinyLM,
        tokenizer: TinyTokenizer,
        global_system_prompt: str = "",
        retriever: WikipediaRetriever | None = None,
        metadata: dict[str, str | int] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.global_system_prompt = global_system_prompt.strip()
        self.retriever = retriever or WikipediaRetriever()
        self.metadata = metadata or {"phase": "unknown", "step": 0, "path": ""}

    @classmethod
    def load(cls, config: ProjectConfig) -> "ChatEngine":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = best_checkpoint(config)
        model, metadata = load_model_with_metadata(checkpoint, device)
        tokenizer = TinyTokenizer.load(config.paths.tokenizer)
        prompt_path = Path(config.paths.system_prompt)
        system_prompt = prompt_path.read_text().strip() if prompt_path.exists() else ""
        return cls(model, tokenizer, system_prompt, metadata=metadata)

    def complete(
        self,
        messages: list[Message | dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        retrieval: bool = False,
    ) -> tuple[str, list[RetrievedPage]]:
        prompt, pages = self.prepare_prompt(messages, max_tokens, retrieval)
        stop_ids = self.stop_ids()
        output = generate(self.model, prompt, max_tokens, temperature, top_p, stop_ids)
        return self.tokenizer.decode(output).strip(), pages

    def prepare_prompt(
        self,
        messages: list[Message | dict[str, str]],
        max_tokens: int,
        retrieval: bool = False,
    ) -> tuple[list[int], list[RetrievedPage]]:
        normalized = normalize_messages(messages)
        system_messages = []
        if self.global_system_prompt:
            system_messages.append(Message("system", self.global_system_prompt))
        system_messages.extend(message for message in normalized if message.role == "system")
        conversation = [message for message in normalized if message.role != "system"]

        pages: list[RetrievedPage] = []
        latest_user = next((message.content for message in reversed(conversation) if message.role == "user"), "")
        if retrieval and latest_user:
            try:
                pages = self.retriever.search(latest_user)
            except requests.RequestException:
                pages = []
            context = format_retrieval_context(pages)
            if context:
                system_messages.append(Message("system", context))

        prompt_budget = max(1, self.model.config.max_seq_len - max_tokens)
        prompt = build_prompt(self.tokenizer, system_messages + conversation, prompt_budget)
        return prompt, pages

    def stop_ids(self) -> set[int]:
        return {
            self.tokenizer.token_id("<|eom|>"),
            self.tokenizer.token_id("<|eot|>"),
            self.tokenizer.token_id("<|user|>"),
            self.tokenizer.token_id("<|system|>"),
        }

    def traced_events(
        self,
        messages: list[Message | dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        retrieval: bool = False,
    ):
        prompt, pages = self.prepare_prompt(messages, max_tokens, retrieval)
        yield {
            "type": "ready",
            "checkpoint": self.metadata,
            "sources": [{"title": page.title, "url": page.url} for page in pages],
        }
        for position, token_id in enumerate(prompt):
            yield {
                "type": "prompt_token",
                "token_id": token_id,
                "token_text": self.tokenizer.token_text(token_id),
                "position": position,
            }

        generated = []
        for event in generate_traced(self.model, prompt, max_tokens, temperature, top_p, self.stop_ids()):
            token_id = event["token_id"]
            generated.append(token_id)
            event["type"] = "token"
            event["token_text"] = self.tokenizer.token_text(token_id)
            event["text"] = self.tokenizer.decode(generated)
            for alternative in event["alternatives"]:
                alternative["token_text"] = self.tokenizer.token_text(alternative["token_id"])
            context_ids = (prompt + generated[:-1])[-self.model.config.max_seq_len :]
            for layer_targets in event["attention_targets"]:
                for target in layer_targets:
                    position = target["position"]
                    target["token_id"] = context_ids[position]
            yield event
        yield {
            "type": "done",
            "text": self.tokenizer.decode(generated).strip(),
            "token_count": len(generated),
        }
