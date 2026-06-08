from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    static = root / "tinyllm" / "static"
    pages = root / "pages"
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(static, output)
    shutil.copytree(pages, output, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static GitHub Pages demo")
    parser.add_argument("--output", type=Path, default=Path("_site"))
    args = parser.parse_args()
    build(args.output)
    print(f"built GitHub Pages site at {args.output}")


if __name__ == "__main__":
    main()
