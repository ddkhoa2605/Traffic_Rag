from __future__ import annotations

from pathlib import Path
from typing import Iterable

import orjson


def write_json(path: Path, value, *, indent: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = orjson.OPT_SORT_KEYS
    if indent:
        options |= orjson.OPT_INDENT_2
    path.write_bytes(orjson.dumps(value, option=options) + b"\n")


def write_jsonl(path: Path, values: Iterable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(orjson.dumps(value, option=orjson.OPT_SORT_KEYS))
            handle.write(b"\n")


def read_json(path: Path):
    return orjson.loads(path.read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        return [orjson.loads(line) for line in handle if line.strip()]

