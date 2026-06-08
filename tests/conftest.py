from __future__ import annotations

import pytest

from tinyllm.tokenizer import TinyTokenizer


@pytest.fixture()
def tokenizer(tmp_path):
    texts = [
        "Hello there. This is a small tokenizer training sentence.",
        "A helpful assistant answers questions clearly and briefly.",
        "The quick brown fox jumps over the lazy dog.",
        "System instructions should remain in the conversation context.",
    ] * 50
    return TinyTokenizer.train(iter(texts), tmp_path / "tokenizer.json", vocab_size=512)

