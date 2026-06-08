FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY tinyllm ./tinyllm
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]

