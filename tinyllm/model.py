from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyllm.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len].to(dtype=x.dtype)[None, None, :, :]
        sin = self.sin[:seq_len].to(dtype=x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project(x)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self._output(output)

    def traced_forward(self, x: torch.Tensor, target_count: int = 3) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
        q, k, v = self._project(x)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        scores = torch.matmul(q[:, :, -1:, :].float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1).mean(dim=(0, 1, 2))
        count = min(target_count, attention.numel())
        weights, positions = torch.topk(attention, count)
        targets = [
            {"position": int(position), "weight": float(weight.detach())}
            for position, weight in zip(positions.cpu(), weights.cpu())
        ]
        return self._output(output), targets

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        repeats = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        return q, k, v

    def _output(self, output: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = output.shape
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))

    def traced_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float, list[dict[str, float | int]]]:
        attention, targets = self.attention.traced_forward(self.attn_norm(x))
        x = x + attention
        x = x + self.feed_forward(self.ffn_norm(x))
        activity = float(x[:, -1, :].float().pow(2).mean().sqrt().detach().cpu())
        return x, activity, targets


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence exceeds maximum length {self.config.max_seq_len}")
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def traced_forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[float], list[list[dict[str, float | int]]]]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence exceeds maximum length {self.config.max_seq_len}")
        x = self.embedding(input_ids)
        layer_activity = []
        attention_targets = []
        for layer in self.layers:
            x, activity, targets = layer.traced_forward(x)
            layer_activity.append(activity)
            attention_targets.append(targets)
        return self.lm_head(self.norm(x)), layer_activity, attention_targets

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)


@torch.inference_mode()
def generate(
    model: TinyLM,
    input_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop_ids: set[int] | None = None,
) -> list[int]:
    device = next(model.parameters()).device
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.max_seq_len :]
        logits, _ = model(context)
        next_token = sample_token(logits[:, -1, :], temperature, top_p)
        token_id = int(next_token.item())
        if stop_ids and token_id in stop_ids:
            break
        generated.append(token_id)
        tokens = torch.cat((tokens, next_token), dim=1)
    return generated


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_ids = probs.sort(descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs[cumulative - sorted_probs > top_p] = 0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids.gather(-1, sampled)


@torch.inference_mode()
def generate_traced(
    model: TinyLM,
    input_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop_ids: set[int] | None = None,
):
    device = next(model.parameters()).device
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    for position in range(max_new_tokens):
        context = tokens[:, -model.config.max_seq_len :]
        logits, layer_activity, attention_targets = model.traced_forward(context)
        next_logits = logits[:, -1, :]
        next_token = sample_token(next_logits, temperature, top_p)
        token_id = int(next_token.item())
        if stop_ids and token_id in stop_ids:
            break

        model_probs = F.softmax(next_logits.float(), dim=-1)[0]
        probability = float(model_probs[token_id].cpu())
        entropy = -(model_probs * model_probs.clamp_min(1e-12).log()).sum() / math.log(model_probs.numel())
        alternative_probs, alternative_ids = torch.topk(model_probs, min(6, model_probs.numel()))
        alternatives = [
            {"token_id": int(candidate_id), "probability": float(candidate_probability)}
            for candidate_id, candidate_probability in zip(alternative_ids.cpu(), alternative_probs.cpu())
            if int(candidate_id) != token_id
        ][:5]
        yield {
            "token_id": token_id,
            "position": position,
            "probability": probability,
            "entropy": float(entropy.cpu()),
            "alternatives": alternatives,
            "layer_activity": layer_activity,
            "attention_targets": attention_targets,
            "context_offset": max(0, tokens.shape[1] - model.config.max_seq_len),
        }
        tokens = torch.cat((tokens, next_token), dim=1)
