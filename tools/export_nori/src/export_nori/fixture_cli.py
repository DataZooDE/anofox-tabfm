"""CLI: uv run make_nori_fixture <out_dir>.

Builds the committed CI fixture (graph + checkpoint + golden file + v2 manifest) and prints the
sha256 of every file.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from export_nori import fixture


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_nori_fixture")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    print(json.dumps(fixture.build(pathlib.Path(args.out)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
