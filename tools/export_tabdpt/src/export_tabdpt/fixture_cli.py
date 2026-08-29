"""CLI: uv run make_tabdpt_fixture <out_dir>.

Builds the committed CI fixture (graphs + safetensors + golden_<task>.json + v2
manifest) and prints the sha256 of every file. Unlike the Orion-BiX fixture this
covers BOTH tasks, and unlike every other fixture it writes ONE safetensors that
both tasks share — TabDPT serves classification and regression from a single
head (see fixture.py).
"""

from __future__ import annotations

import argparse
import json
import pathlib

from export_tabdpt import fixture


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_tabdpt_fixture")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    hashes = fixture.build(pathlib.Path(args.out))
    print(json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
