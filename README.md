# Tiny Conversational LLM

A roughly 32M-parameter, English-only conversational language model that trains
from random weights and a new byte-level BPE tokenizer. It is intentionally
small enough to train on an RTX 4060 with 8 GB VRAM.

The quick profile limits pretraining to 90 minutes and instruction tuning to 15
minutes. At that budget this is an educational, experimental assistant, not a
replacement for a pretrained production model. Optional live Wikipedia
retrieval gives it access to newer facts without baking them into its weights.

## Requirements

- NVIDIA GPU with recent drivers and the NVIDIA Container Toolkit
- Docker and the NVIDIA Container Toolkit
- About 35 GB free disk for caches, packed data, and checkpoints

The host's Python environment is not used. The container uses Python and CUDA
versions known to work together.

## Train It

```bash
make build
make data
make train-quick
make sft-quick
make chat
```

`make data` is a one-time preparation step and is not included in the training
time limit. It streams source datasets, trains the tokenizer, and writes compact
`uint16` token shards under `data/packed/`. It does not download the full
FineWeb-Edu dataset.

Both training commands resume their latest checkpoint automatically. They save
regularly and save once more when interrupted or when their wall-clock limit is
reached. Change model, data, and time limits in `configs/quick.json`.

## Chat And Instruct

Copy `system_prompt.example.txt` to the ignored local `system_prompt.txt`, then
edit it to set the backend-wide instruction. Request-specific system messages
are added after it.

```bash
cp system_prompt.example.txt system_prompt.txt
```

Terminal chat supports `/retrieval on`, `/retrieval off`, `/clear`, and `/quit`:

```bash
make chat
```

Start the HTTP API:

```bash
make serve
```

This also starts the 3D token-brain visualizer. Open
[http://localhost:8000](http://localhost:8000) after the server starts. The
first load builds and caches a constellation from the loaded checkpoint's token
embeddings. You can rotate and zoom it, hover over tokens, and watch prompt and
answer paths form as the model generates.

The visualizer reports real model signals:

- Cyan nodes and paths are prompt tokens; amber nodes and paths are generated tokens.
- The central stack pulses with residual activation strength across transformer layers.
- Purple links summarize the strongest attention targets for the latest token.
- The side panel shows selected-token confidence, vocabulary uncertainty, and alternatives.

These signals are useful views into generation, but they are not a literal
English transcript of the model's thoughts. The renderer is bundled locally and
works without internet access. Avoid running the visualizer alongside training,
because both processes compete for GPU memory and compute.

The equivalent explicit command is:

```bash
make visualize
```

When using the local virtual environment instead of Docker:

```bash
.venv/bin/python -m tinyllm.api --config configs/quick.json
```

Stop training before starting the visualizer, then open
[http://localhost:8000](http://localhost:8000).

## Publish The Static Demo

GitHub Pages can host static files, but it cannot run PyTorch, load the model
checkpoint, or provide the WebSocket backend. The public Pages site therefore
replays a recorded trace over a sanitized copy of the real token-embedding
layout. The local FastAPI site remains the live, promptable version.

The repository includes a GitHub Actions Pages workflow. Before pushing:

```bash
make pages
python3 -m http.server --directory _site 8080
```

Open [http://localhost:8080](http://localhost:8080) to preview exactly what
GitHub Pages will publish. Then create a GitHub repository, push this project to
its `main` branch, and choose **GitHub Actions** as the Pages source in the
repository's **Settings → Pages** screen. Every push to `main` will rebuild and
deploy the demo.

Large and private local files are excluded by `.gitignore`, including
checkpoints, training datasets, virtual environments, caches, logs, and the
generated `_site/` directory. The tracked `pages/data/` files contain only the
sanitized public layout and recorded demo events.

The frontend also supports a separately hosted live backend by changing
`pages/site-config.json` to live mode and setting `backend_url`. When doing
that, set `TINYLLM_ALLOWED_ORIGINS` on the backend to the exact Pages origin.
Do not expose an unauthenticated GPU backend publicly without rate limiting.

To replace the public replay after training a new model:

```bash
.venv/bin/python scripts/export_pages_demo.py \
  --layout artifacts/visualizer/layout-REPLACE_ME.json \
  --tokenizer artifacts/tokenizer.json
```

Then call the OpenAI-style, non-streaming endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "Answer in one sentence."},
      {"role": "user", "content": "What is a transformer model?"}
    ],
    "retrieval": false,
    "max_tokens": 80
  }'
```

Set `"retrieval": true` to search live English Wikipedia. Retrieved source URLs
are returned in the response's `sources` field.

## Data

- Pretraining: [FineWeb-Edu `sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  streamed until the configured token cap. License: ODC-By-1.0 and subject to
  Common Crawl terms.
- Instruction tuning: [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
  and top-ranked English branches from [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1).
  Both are Apache-2.0.

Each prepared dataset includes `data-manifest.json` with source URLs, resolved
dataset revisions, licenses, counts, and preparation settings.

## Useful Commands

```bash
make test       # unit tests plus tiny end-to-end training smoke test
make smoke      # only the tiny end-to-end training test
```

Equivalent local commands are available after installing `.[dev]`, for example
`python -m tinyllm.train pretrain --config configs/quick.json`.
