"""CLI: uv run make_orion_bix_fixture <out_dir>.

Builds the committed CI fixture (graph + safetensors + golden.json + v2
manifest) and prints the sha256 of every file. Orion-BiX is classification-only,
so there is no --task flag.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from export_orion_bix import fixture


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_orion_bix_fixture")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    hashes = fixture.build(pathlib.Path(args.out))
    print(json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
