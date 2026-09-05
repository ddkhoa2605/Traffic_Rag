from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental directed legal graph API")
    parser.add_argument("--root", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9630)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install graph dependencies: pip install -e ".[graph]"') from exc
    root = Path(args.root or Path.cwd()).resolve()
    from src.graph.api import create_app

    uvicorn.run(create_app(root), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
