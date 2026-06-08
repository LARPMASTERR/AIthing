import requests
import torch

from tinyllm.config import ModelConfig
from tinyllm.engine import ChatEngine
from tinyllm.model import TinyLM


class BrokenRetriever:
    def search(self, query):
        raise requests.RequestException("offline")


def test_retrieval_failure_falls_back(tokenizer):
    model = TinyLM(
        ModelConfig(
            vocab_size=512,
            max_seq_len=64,
            n_layers=1,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
        )
    ).eval()
    engine = ChatEngine(model, tokenizer, "Global instruction", BrokenRetriever())
    text, pages = engine.complete(
        [{"role": "system", "content": "Request instruction"}, {"role": "user", "content": "Hello"}],
        max_tokens=2,
        temperature=0,
        retrieval=True,
    )
    assert isinstance(text, str)
    assert pages == []
