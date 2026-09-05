from __future__ import annotations

import argparse
from pathlib import Path

from src.demo_web.app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the read-only Traffic Law RAG demo WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9640)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Thiếu web dependencies. Chạy: python -m pip install -e ".[demo]"') from exc
    root = Path(args.root or Path.cwd()).resolve()
    uvicorn.run(create_app(root), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

