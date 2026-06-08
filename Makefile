IMAGE = tiny-convo-llm
RUN = docker run --rm --gpus all -v $(CURDIR):/workspace -w /workspace -e HF_HOME=/workspace/data/huggingface $(IMAGE)

.PHONY: build data train-quick sft-quick serve visualize chat pages test smoke

build:
	docker build -t $(IMAGE) .

data:
	$(RUN) python -m tinyllm.prepare all --config configs/quick.json

train-quick:
	$(RUN) python -m tinyllm.train pretrain --config configs/quick.json

sft-quick:
	$(RUN) python -m tinyllm.train sft --config configs/quick.json

serve:
	docker run --rm --gpus all -p 8000:8000 -v $(CURDIR):/workspace -w /workspace -e HF_HOME=/workspace/data/huggingface $(IMAGE) python -m tinyllm.api --config configs/quick.json

visualize: serve

chat:
	$(RUN) python -m tinyllm.cli chat --config configs/quick.json

pages:
	python3 scripts/build_pages.py --output _site

test:
	$(RUN) pytest -q

smoke:
	$(RUN) pytest -q tests/test_smoke.py
