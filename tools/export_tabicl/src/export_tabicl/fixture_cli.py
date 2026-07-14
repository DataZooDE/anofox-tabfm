"""CLI: uv run make_tabicl_fixture <out_dir> [--task classification|regression].

Builds the committed CI fixture (graph + safetensors + golden.json + v2
manifest) and prints the sha256 of every file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from export_tabicl import fixture


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_tabicl_fixture")
    ap.add_argument("out")
    ap.add_argument("--task", default="classification",
                    choices=["classification", "regression"])
    args = ap.parse_args(argv)
    hashes = fixture.build(pathlib.Path(args.out), task=args.task)
    print(json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
