"""CLIs: `uv run make_fixture --out DIR` and `uv run golden_preprocess`."""

from __future__ import annotations

import argparse
import json
import pathlib


def main_fixture(argv=None) -> int:
  from make_fixture import fixture

  ap = argparse.ArgumentParser(prog="make_fixture")
  ap.add_argument("--out", required=True,
                  help="output dir (the repo's test/fixtures)")
  ap.add_argument("--task", default="classification",
                  choices=("classification", "regression"),
                  help="fixture task (regression -> logits [1, T, 1])")
  ap.add_argument("--skip-roundtrip", action="store_true")
  args = ap.parse_args(argv)
  out = pathlib.Path(args.out)

  hashes = fixture.build(out, task=args.task)

  if not args.skip_roundtrip:
    rel = fixture.roundtrip_check(out)
    print(f"[make_fixture] golden roundtrip (inject -> ORT): max rel delta "
          f"{rel:.2e} (budget {fixture.PARITY_RTOL:.0e})")
    assert rel < fixture.PARITY_RTOL, "golden roundtrip out of budget"

  # sha256sum -c compatible pin file; CI verifies, never regenerates.
  pin = "".join(f"{sha}  {name}\n" for name, sha in sorted(hashes.items()))
  (out / "FIXTURE_SHA256").write_text(pin)

  total = 0
  for name in fixture.FILES:
    size = (out / name).stat().st_size
    total += size
    print(f"[make_fixture] {name}: {size:,} B  sha256 {hashes[name]}")
  print(f"[make_fixture] total {total:,} B (< 5 MB budget)")
  print(f"[make_fixture] pinned safetensors sha256: "
        f"{hashes['model.safetensors']}")
  return 0


def main_golden_preprocess(argv=None) -> int:
  from make_fixture import golden_preprocess

  ap = argparse.ArgumentParser(prog="golden_preprocess")
  ap.add_argument("--out", required=True,
                  help="output json path (test/fixtures/golden_preprocess.json)")
  args = ap.parse_args(argv)
  out = pathlib.Path(args.out)
  payload = golden_preprocess.generate(out)
  print(f"[golden_preprocess] wrote {out} ({out.stat().st_size:,} B, "
        f"{len(payload['tables'])} tables, "
        f"{len(payload['ensemble']['cases'])} ensemble cases)")
  json.loads(out.read_text())  # self-check: valid JSON

  # keep the pin file covering this artifact too (update-or-append)
  pin_path = out.parent / "FIXTURE_SHA256"
  if pin_path.exists():
    import hashlib
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = [l for l in pin_path.read_text().splitlines()
             if not l.endswith(f"  {out.name}")]
    lines.append(f"{sha}  {out.name}")
    pin_path.write_text("\n".join(sorted(lines, key=lambda l: l.split()[1]))
                        + "\n")
    print(f"[golden_preprocess] pinned in {pin_path}: {sha}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main_fixture())
