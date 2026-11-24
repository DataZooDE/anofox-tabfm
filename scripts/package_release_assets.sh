#!/usr/bin/env bash
# Package the CPU extension binaries a pipeline run built into GitHub release assets.
#
#   scripts/package_release_assets.sh <artifacts dir> <out dir> [extension name]
#
# <artifacts dir> holds one directory per build artifact, named as extension-ci-tools names them,
# "<ext>-<duckdb version>-extension-<platform>", each with a single <ext>.duckdb_extension inside (what
# actions/download-artifact writes for a pattern download). For every one, <out dir> gets
#   <ext>-<duckdb version>-<platform>.duckdb_extension.gz
# and one SHA256SUMS over all of them. These are the same binaries DuckDB's extension server would serve, but UNSIGNED:
# load them with `duckdb -unsigned` (or SET allow_unsigned_extensions = true) and LOAD '<path>'. Signed builds come only
# from `INSTALL anofox_tabfm FROM community`.
#
# Fails when no artifact is found or one holds no extension binary: a release must never silently ship without them.
set -euo pipefail

in_dir=${1:?usage: $0 <artifacts dir> <out dir> [extension name]}
out_dir=${2:?usage: $0 <artifacts dir> <out dir> [extension name]}
ext=${3:-anofox_tabfm}
mkdir -p "$out_dir"

count=0
shopt -s nullglob
for dir in "$in_dir"/"$ext"-v*-extension-*; do
	[ -d "$dir" ] || continue
	name=$(basename "$dir")
	case "$name" in *.wasm) continue ;; esac
	rest=${name#"$ext"-}             # v1.5.6-extension-linux_amd64
	duckdb_version=${rest%%-extension-*}
	platform=${rest#*-extension-}
	bin="$dir/$ext.duckdb_extension"
	if [ ! -f "$bin" ]; then
		echo "error: $name holds no $ext.duckdb_extension" >&2
		exit 1
	fi
	asset="$ext-$duckdb_version-$platform.duckdb_extension.gz"
	gzip -9 -n -c "$bin" > "$out_dir/$asset"
	echo "$asset ($(wc -c < "$out_dir/$asset") bytes)"
	count=$((count + 1))
done
if [ "$count" -eq 0 ]; then
	echo "error: no $ext-v*-extension-* artifact under $in_dir" >&2
	exit 1
fi
(cd "$out_dir" && sha256sum -- "$ext"-*.duckdb_extension.gz > SHA256SUMS)
echo "$count binaries, SHA256SUMS written"
