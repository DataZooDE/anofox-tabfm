"""CLI: uv run make_causilo_fixture <out_dir>.

Builds the committed CI fixture (graphs + safetensors + golden files + v2 manifest) and
prints the sha256 of every file.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from export_causilo import fixture


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_causilo_fixture")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    print(json.dumps(fixture.build(pathlib.Path(args.out)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
