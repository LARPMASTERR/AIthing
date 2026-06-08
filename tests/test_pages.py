from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_branch_pages_fallback_opens_browser_app():
    root = Path(__file__).resolve().parents[1]
    assert "tinyllm/static/index.html?config=../../site-config.json" in (root / "index.html").read_text()
    config = json.loads((root / "site-config.json").read_text())
    assert config["mode"] == "browser"
    assert config["model_url"] == "../../pages/model/tinyllm.onnx"


def test_pages_build_contains_browser_model_and_is_sanitized(tmp_path):
    output = tmp_path / "site"
    subprocess.run(
        [sys.executable, "scripts/build_pages.py", "--output", str(output)],
        check=True,
    )
    required = [
        "index.html",
        "app.js",
        "style.css",
        "site-config.json",
        "data/layout.json",
        "model/config.json",
        "model/tokenizer.json",
        "model/tinyllm.onnx",
        "browser-model.js",
        "tiny-tokenizer.js",
        "vendor/three.module.min.js",
        "vendor/three.core.min.js",
        "vendor/OrbitControls.js",
        "vendor/ort/ort.all.bundle.min.mjs",
        "vendor/ort/ort-wasm-simd-threaded.jsep.mjs",
        "vendor/ort/ort-wasm-simd-threaded.jsep.wasm",
    ]
    assert all((output / path).exists() for path in required)
    config = json.loads((output / "site-config.json").read_text())
    assert config["mode"] == "browser"
    assert config["model_url"] == "./model/tinyllm.onnx"
    assert (output / "model/tinyllm.onnx").stat().st_size < 100 * 1024 * 1024
    assert not (output / "data/demo-events.json").exists()
    assert json.loads((output / "data/layout.json").read_text())["checkpoint"] == {
        "phase": "sft",
        "step": 1391,
    }
    html = (output / "index.html").read_text()
    assert 'href="./style.css"' in html
    assert 'src="./app.js"' in html
    text_suffixes = {".css", ".html", ".js", ".json", ".md", ".mjs", ".txt", ".yml"}
    public_text = "\n".join(
        path.read_text(errors="ignore")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in text_suffixes and "vendor" not in path.parts
    )
    assert "/home/" not in public_text
    assert "artifacts/checkpoints" not in public_text
