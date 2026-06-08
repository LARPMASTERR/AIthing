from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pages_build_is_static_and_sanitized(tmp_path):
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
        "data/demo-events.json",
        "vendor/three.module.min.js",
        "vendor/three.core.min.js",
        "vendor/OrbitControls.js",
    ]
    assert all((output / path).exists() for path in required)
    assert json.loads((output / "site-config.json").read_text())["mode"] == "demo"
    assert json.loads((output / "data/layout.json").read_text())["checkpoint"]["phase"] == "public-demo"
    html = (output / "index.html").read_text()
    assert 'href="./style.css"' in html
    assert 'src="./app.js"' in html
    public_text = "\n".join(
        path.read_text(errors="ignore")
        for path in output.rglob("*")
        if path.is_file()
    )
    assert "/home/" not in public_text
    assert "artifacts/checkpoints" not in public_text
