from __future__ import annotations

import argparse

from tinyllm.config import ProjectConfig
from tinyllm.engine import ChatEngine


def chat(config_path: str, retrieval: bool) -> None:
    engine = ChatEngine.load(ProjectConfig.load(config_path))
    messages: list[dict[str, str]] = []
    print("TinyLM chat. Commands: /clear, /retrieval on, /retrieval off, /quit")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text == "/quit":
            return
        if text == "/clear":
            messages.clear()
            print("conversation cleared")
            continue
        if text in {"/retrieval on", "/retrieval off"}:
            retrieval = text.endswith("on")
            print(f"retrieval {'enabled' if retrieval else 'disabled'}")
            continue
        messages.append({"role": "user", "content": text})
        response, pages = engine.complete(messages, retrieval=retrieval)
        messages.append({"role": "assistant", "content": response})
        print(f"\nassistant> {response}")
        for page in pages:
            print(f"source> {page.url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny conversational LLM tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--config", default="configs/quick.json")
    chat_parser.add_argument("--retrieval", action="store_true")
    args = parser.parse_args()
    if args.command == "chat":
        chat(args.config, args.retrieval)


if __name__ == "__main__":
    main()

