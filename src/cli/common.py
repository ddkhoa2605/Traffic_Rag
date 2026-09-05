from __future__ import annotations

import argparse

from src.registry.loader import load_registry


def document_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("document_id", choices=["LAW_35_2024", "LAW_36_2024"])
    parser.add_argument("--root", default=None, help="Repository root (defaults to current directory)")
    return parser


def registry_from(args):
    return load_registry(args.root)

