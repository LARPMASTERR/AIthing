from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|eom|>",
    "<|eot|>",
]


class TinyTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        path: str | Path,
        vocab_size: int = 16_384,
    ) -> "TinyTokenizer":
        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(path))
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str | Path) -> "TinyTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"tokenizer is missing required token {token}")
        return token_id

    def token_text(self, token_id: int) -> str:
        text = self.decode([token_id], skip_special_tokens=False)
        if text:
            return text
        token = self.tokenizer.id_to_token(token_id)
        return token or ""

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()
