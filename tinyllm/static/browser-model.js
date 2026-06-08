import * as ort from "./vendor/ort/ort.all.bundle.min.mjs";
import { TinyTokenizer } from "./tiny-tokenizer.js";

function softmax(logits, temperature = 1) {
  let maximum = -Infinity;
  for (const value of logits) maximum = Math.max(maximum, value / temperature);
  const probabilities = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    probabilities[index] = Math.exp(logits[index] / temperature - maximum);
    total += probabilities[index];
  }
  for (let index = 0; index < probabilities.length; index += 1) probabilities[index] /= total;
  return probabilities;
}

function ranked(probabilities) {
  return Array.from(probabilities, (probability, tokenId) => ({ tokenId, probability }))
    .sort((left, right) => right.probability - left.probability);
}

function sample(logits, temperature, topP) {
  if (temperature <= 0) {
    let best = 0;
    for (let index = 1; index < logits.length; index += 1) {
      if (logits[index] > logits[best]) best = index;
    }
    return best;
  }
  const candidates = ranked(softmax(logits, temperature));
  let limit = 1;
  let cumulative = candidates[0].probability;
  while (limit < candidates.length && cumulative < topP) {
    cumulative += candidates[limit].probability;
    limit += 1;
  }
  let target = Math.random() * cumulative;
  for (let index = 0; index < limit; index += 1) {
    target -= candidates[index].probability;
    if (target <= 0) return candidates[index].tokenId;
  }
  return candidates[limit - 1].tokenId;
}

function formatMessage(message) {
  return `<|${message.role}|>\n${message.content.trim()}<|eom|>\n`;
}

export class BrowserModel {
  constructor(session, tokenizer, config) {
    this.session = session;
    this.tokenizer = tokenizer;
    this.config = config;
    this.stopIds = new Set([
      tokenizer.tokenId(config.special_tokens.eom),
      tokenizer.tokenId(config.special_tokens.eot),
      tokenizer.tokenId(config.special_tokens.user),
      tokenizer.tokenId(config.special_tokens.system),
    ]);
  }

  static async load(siteConfig, status) {
    ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
    ort.env.wasm.numThreads = 1;
    const [tokenizer, config] = await Promise.all([
      TinyTokenizer.load(siteConfig.tokenizer_url),
      fetch(siteConfig.model_config_url).then((response) => {
        if (!response.ok) throw new Error(`model config download failed (${response.status})`);
        return response.json();
      }),
    ]);
    let providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    status(`Loading ${config.checkpoint.phase} model with ${providers[0].toUpperCase()}...`);
    let session;
    try {
      session = await ort.InferenceSession.create(siteConfig.model_url, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
    } catch (error) {
      if (!navigator.gpu) throw error;
      providers = ["wasm"];
      status("WebGPU could not load this model. Falling back to WASM...");
      session = await ort.InferenceSession.create(siteConfig.model_url, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
    }
    config.execution_provider = providers.join(" / ");
    return new BrowserModel(session, tokenizer, config);
  }

  buildPrompt(messages, maxNewTokens) {
    const system = [
      { role: "system", content: this.config.default_system_prompt },
      ...messages.filter((message) => message.role === "system"),
    ];
    const conversation = messages.filter((message) => message.role !== "system");
    const assistant = this.tokenizer.encode(`${this.config.special_tokens.assistant}\n`);
    const budget = Math.max(1, this.config.max_seq_len - maxNewTokens);
    const latestUser = conversation.findLast((message) => message.role === "user");
    const required = [...system, ...(latestUser ? [latestUser] : [])];
    const requiredParts = required.map((message) => ({
      message,
      prefix: this.tokenizer.encode(`<|${message.role}|>\n`),
      content: this.tokenizer.encode(message.content.trim()),
      suffix: this.tokenizer.encode("<|eom|>\n"),
    }));
    const overhead = assistant.length + requiredParts.reduce(
      (total, part) => total + part.prefix.length + part.suffix.length,
      0,
    );
    let contentBudget = Math.max(0, budget - overhead);
    const contentLimits = requiredParts.map(() => 0);
    while (contentBudget > 0 && requiredParts.some((part, index) => contentLimits[index] < part.content.length)) {
      for (let index = 0; index < requiredParts.length && contentBudget > 0; index += 1) {
        if (contentLimits[index] >= requiredParts[index].content.length) continue;
        contentLimits[index] += 1;
        contentBudget -= 1;
      }
    }
    const requiredTokens = requiredParts.map((part, index) => {
      const limit = contentLimits[index];
      const content = part.message === latestUser && limit
        ? part.content.slice(-limit)
        : part.content.slice(0, limit);
      return [...part.prefix, ...content, ...part.suffix];
    });
    let historyTokens = [];
    for (let index = conversation.length - 1; index >= 0; index -= 1) {
      if (conversation[index] === latestUser) continue;
      const encoded = this.tokenizer.encode(formatMessage(conversation[index]));
      const requiredLength = requiredTokens.reduce((total, tokens) => total + tokens.length, 0);
      if (requiredLength + assistant.length + historyTokens.length + encoded.length > budget) continue;
      historyTokens = [...encoded, ...historyTokens];
    }
    const systemTokens = requiredTokens.slice(0, system.length).flat();
    const latestTokens = latestUser ? requiredTokens.at(-1) : [];
    return [...systemTokens, ...historyTokens, ...latestTokens, ...assistant].slice(-budget);
  }

  async *generate(messages, options, shouldStop) {
    const prompt = this.buildPrompt(messages, options.maxTokens);
    yield { type: "ready", checkpoint: this.config.checkpoint, sources: [] };
    for (let position = 0; position < prompt.length; position += 1) {
      const tokenId = prompt[position];
      yield {
        type: "prompt_token",
        token_id: tokenId,
        token_text: this.tokenizer.tokenText(tokenId),
        position,
      };
    }

    const generated = [];
    let tokens = [...prompt];
    for (let position = 0; position < options.maxTokens && !shouldStop(); position += 1) {
      const context = tokens.slice(-this.config.max_seq_len);
      const input = new ort.Tensor(
        "int64",
        BigInt64Array.from(context, (tokenId) => BigInt(tokenId)),
        [1, context.length],
      );
      const output = await this.session.run({ input_ids: input });
      if (shouldStop()) break;
      const logits = output.logits.data;
      const tokenId = sample(logits, options.temperature, options.topP);
      if (this.stopIds.has(tokenId)) break;

      const probabilities = softmax(logits);
      const alternatives = ranked(probabilities)
        .filter((candidate) => candidate.tokenId !== tokenId)
        .slice(0, 5)
        .map((candidate) => ({
          token_id: candidate.tokenId,
          token_text: this.tokenizer.tokenText(candidate.tokenId),
          probability: candidate.probability,
        }));
      let entropy = 0;
      for (const probability of probabilities) {
        if (probability > 0) entropy -= probability * Math.log(probability);
      }
      const attention = output.attention.data;
      const attentionTargets = [];
      for (let layer = 0; layer < this.config.n_layers; layer += 1) {
        const targets = [];
        for (let contextPosition = 0; contextPosition < context.length; contextPosition += 1) {
          targets.push({
            position: contextPosition,
            token_id: context[contextPosition],
            weight: attention[layer * context.length + contextPosition],
          });
        }
        targets.sort((left, right) => right.weight - left.weight);
        attentionTargets.push(targets.slice(0, 3));
      }
      generated.push(tokenId);
      tokens.push(tokenId);
      yield {
        type: "token",
        token_id: tokenId,
        token_text: this.tokenizer.tokenText(tokenId),
        text: this.tokenizer.decode(generated),
        position,
        probability: probabilities[tokenId],
        entropy: entropy / Math.log(probabilities.length),
        alternatives,
        layer_activity: Array.from(output.layer_activity.data),
        attention_targets: attentionTargets,
        context_offset: Math.max(0, tokens.length - this.config.max_seq_len),
      };
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const text = this.tokenizer.decode(generated).trim();
    yield { type: "done", text, token_count: generated.length };
  }
}
