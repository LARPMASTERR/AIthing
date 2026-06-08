from fastapi.testclient import TestClient

from tinyllm.api import create_app
from tinyllm.retrieval import RetrievedPage


class FakeEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metadata = {"phase": "test", "step": 4, "path": "test.pt"}

    def complete(self, messages, max_tokens, temperature, top_p, retrieval):
        pages = [RetrievedPage("Current fact", "extract", "https://example.com")] if retrieval else []
        return "hello", pages

    def traced_events(self, messages, max_tokens, temperature, top_p, retrieval):
        yield {"type": "ready", "checkpoint": self.metadata, "sources": []}
        yield {"type": "prompt_token", "token_id": 1, "token_text": "Hi", "position": 0}
        yield {
            "type": "token",
            "token_id": 2,
            "token_text": "hello",
            "text": "hello",
            "position": 0,
            "probability": 0.5,
            "entropy": 0.4,
            "alternatives": [],
            "layer_activity": [0.1],
            "attention_targets": [[]],
        }
        yield {"type": "done", "text": "hello", "token_count": 1}


def test_chat_api_and_sources(tokenizer):
    client = TestClient(create_app(FakeEngine(tokenizer)))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "retrieval": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["sources"][0]["url"] == "https://example.com"


def test_chat_api_validation(tokenizer):
    client = TestClient(create_app(FakeEngine(tokenizer)))
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "bad", "content": "Hi"}]})
    assert response.status_code == 422


def test_visualizer_static_route_and_websocket_events(tokenizer):
    client = TestClient(create_app(FakeEngine(tokenizer)))
    assert client.get("/").status_code == 200
    for asset in ["app.js", "vendor/three.module.min.js", "vendor/three.core.min.js", "vendor/OrbitControls.js"]:
        assert client.get(f"/static/{asset}").status_code == 200
    with client.websocket_connect("/ws/visualize") as socket:
        socket.send_json({"messages": [{"role": "user", "content": "Hi"}]})
        events = [socket.receive_json() for _ in range(4)]
    assert [event["type"] for event in events] == ["ready", "prompt_token", "token", "done"]
