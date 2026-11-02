#!/usr/bin/env python3
"""Rename anofox-tabfm cache directories written under "@main" to the revisions the registry now pins.

Since this release every built-in model downloads its Hugging Face weights from a fixed commit, and the cache
directory is named after it: "<owner>__<repo>@<commit>" instead of "<owner>__<repo>@main". A cache written by an
earlier release is therefore not found, and the next tabfm_download fetches the same bytes again. This script
renames each "@main" directory instead, but only when every weights file the registry declares that is present
in it has exactly the pinned size (and, with --sha256, the pinned upstream LFS sha256). Converted GPU files,
graphs and anything else in the directory move with it. Nothing is deleted, and an existing pinned directory
is never overwritten.

Dry run by default; pass --apply to rename.

    python3 tools/migrate_cache_revisions.py                  # what would move, size check
    python3 tools/migrate_cache_revisions.py --sha256         # also hash every file (slow for tabfm-v1, 13 GB)
    python3 tools/migrate_cache_revisions.py --apply [--cache-dir DIR]

The table below mirrors src/tabfm_registry.cpp and src/tabfm_manifest.cpp (repo -> pinned commit, and per
cache-relative file: bytes, LFS sha256 at that commit).
"""
import argparse
import hashlib
import pathlib
import sys

PINS = {
    "Layer6/TabDPT": ("a5ca6e01c0fa09ec68c73e958e5199d1932abb3a", [
        ("model.safetensors", 254098072, "06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"),
        ("tabdpt1_3/model.safetensors", 252233296, "97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb"),
    ]),
    "Lexsi/Orion-BiX": ("9504590d769648eeeb7ecfb0fead2a01c11fb12a", [
        ("classification/model.ckpt", 314653437, "c53afd272e05e9bd6740698b6fa6f3d8789fc4f6da263415d3287b9ed51cd979"),
    ]),
    "Lexsi/Orion-MSP": ("8b712f6ff699750f7ac5825f31e89e6f6f161577", [
        ("classification/model.ckpt", 335545028, "7f438b5d7321d23a531dfef5ab8edab7c474c75b500abfcb318e25ad32d5f9de"),
    ]),
    "Prior-Labs/TabPFN-v2-clf": ("275f19d2add9c2c0de38f9ebaded3856599fa2e7", [
        ("classification/model.ckpt", 29016968, "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f"),
    ]),
    "Prior-Labs/TabPFN-v2-reg": ("02a44705ff7790b50cded404bc59adb94da456a1", [
        ("regression/model.ckpt", 44390977, "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"),
    ]),
    "Prior-Labs/tabpfn_2_5": ("6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b", [
        ("classification-real/model.ckpt", 42929707, "b403ee3947766675a0a7c2fd9c31ee41a9c8d080e8b5e30bb31707ed744be267"),
        ("classification/model.ckpt", 42935499, "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12"),
        ("regression-real/model.ckpt", 40831868, "8ab42d2d0abe3886a2c54a336b2abf251658a23a9b3f6559c4fdf0c024f48e63"),
        ("regression/model.ckpt", 40831995, "6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8"),
    ]),
    "Prior-Labs/tabpfn_2_6": ("24148873a9a5992b429833d07363690dcffd02a8", [
        ("classification/model.ckpt", 43044699, "0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a"),
        ("regression/model.ckpt", 51576286, "c855169116d79d2c957a29107ffc04464555f39aec0eef90c90390f2c2b2008b"),
    ]),
    "Prior-Labs/tabpfn_3": ("24a16a89d245878b846555110985634aa2e656d7", [
        ("classification/model.ckpt", 212804803, "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"),
        ("regression/model.ckpt", 233289807, "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"),
    ]),
    "Prior-Labs/tabpfn_3_5": ("06bf2ba35c80a92a3b9abb436b99cf49e7a0365e", [
        ("v3.5-fast/model.safetensors", 334181708, "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550"),
        ("v3.5/model.safetensors", 876027932, "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"),
    ]),
    "autogluon/mitra-classifier": ("c425e9fa0910a6be1c494321792e7ba2a1367b1a", [
        ("model.safetensors", 302717904, "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"),
    ]),
    "autogluon/mitra-classifier-2": ("edada0d20759c58ada8c8605c25f22f6e98ea5f0", [
        ("model.safetensors", 302717904, "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09"),
    ]),
    "autogluon/mitra-regressor": ("5f277aa8f69042d39d6ac3612aed18bb9279bd95", [
        ("model.safetensors", 302683140, "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"),
    ]),
    "autogluon/mitra-regressor-2": ("246c7d310fff04296a2388df5f448e3ffe809283", [
        ("model.safetensors", 306776952, "8c90b8f281a793f9feb19f8fc3c5529c58ac844dd7ffc690a0fdbb96fae0967c"),
    ]),
    "google/tabfm-1.0.0-pytorch": ("77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83", [
        ("classification/model.safetensors", 6557888408, "928cb350becdc77cdb7a9e8c36deda88917bfd14a3091894a2dc516db58a2085"),
        ("regression/model.safetensors", 6591243724, "bd5a615b0322a8f04a895038de6df6fbd71430eca750e1d792f31048654674a9"),
    ]),
    "jingang/TabICL": ("4dcd344ece2c00be9e831fdd35bed57b5ad83e19", [
        ("classification/model.ckpt", 110368038, "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"),
        ("regression/model.ckpt", 114324594, "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"),
    ]),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", default="~/.cache/anofox-tabfm",
                    help="the anofox_tabfm_cache_dir (default ~/.cache/anofox-tabfm)")
    ap.add_argument("--apply", action="store_true", help="rename (default: only report)")
    ap.add_argument("--sha256", action="store_true", help="also compare each present file's sha256")
    args = ap.parse_args(argv)
    cache = pathlib.Path(args.cache_dir).expanduser()
    if not cache.is_dir():
        print(f"no cache directory at {cache}")
        return 0
    status = 0
    for repo, (revision, files) in sorted(PINS.items()):
        slug = repo.replace("/", "__")
        old, new = cache / f"{slug}@main", cache / f"{slug}@{revision}"
        if not old.is_dir():
            continue
        if new.exists():
            print(f"skip   {old.name}: {new.name} already exists (left both alone)")
            continue
        problems, checked = [], 0
        for rel, size, oid in files:
            f = old / rel
            if not f.is_file():
                continue
            if f.stat().st_size != size:
                problems.append(f"{rel} is {f.stat().st_size} bytes, pinned {size}")
            elif args.sha256 and sha256(f) != oid:
                problems.append(f"{rel} sha256 differs from the pinned upload")
            else:
                checked += 1
        if problems:
            status = 1
            print(f"keep   {old.name}: not the pinned upload ({'; '.join(problems)}); tabfm_download fetches it again")
            continue
        how = "sha256" if args.sha256 else "size"
        note = f"{checked} weights file(s) match by {how}" if checked else "no declared weights file present (converted files only)"
        if args.apply:
            old.rename(new)
            print(f"moved  {old.name} -> {new.name} ({note})")
        else:
            print(f"would  {old.name} -> {new.name} ({note})")
    if not args.apply:
        print("dry run: nothing changed; pass --apply to rename")
    return status


if __name__ == "__main__":
    sys.exit(main())
