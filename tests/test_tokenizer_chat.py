from tinyllm.chat import Message, build_prompt, encode_sft, format_messages
from tinyllm.data import deterministic_split
from tinyllm.tokenizer import SPECIAL_TOKENS


def test_tokenizer_round_trip_and_special_tokens(tokenizer):
    text = "Hello, byte-level tokenizer!\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert all(tokenizer.token_id(token) >= 0 for token in SPECIAL_TOKENS)


def test_sft_masks_only_assistant_content(tokenizer):
    messages = [Message("user", "Question"), Message("assistant", "Answer")]
    tokens, mask = encode_sft(tokenizer, messages)
    user_length = len(tokenizer.encode(format_messages(messages[:1])))
    assistant_prefix = len(tokenizer.encode("<|assistant|>\n"))
    assert not any(mask[: user_length + assistant_prefix])
    assert all(mask[user_length + assistant_prefix :])
    assert len(tokens) == len(mask)


def test_prompt_keeps_system_and_latest_user_when_truncated(tokenizer):
    messages = [
        Message("system", "KEEP_SYSTEM " * 20),
        Message("user", "old message " * 20),
        Message("assistant", "old answer " * 20),
        Message("user", "KEEP_LATEST " * 20),
    ]
    prompt = tokenizer.decode(build_prompt(tokenizer, messages, 80), skip_special_tokens=False)
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "KEEP_SYSTEM" in prompt
    assert "KEEP_LATEST" in prompt


def test_deterministic_split():
    assert deterministic_split("same-key") == deterministic_split("same-key")

