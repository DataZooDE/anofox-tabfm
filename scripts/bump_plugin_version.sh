#!/usr/bin/env bash
# Set the Claude Code plugin's version to a release (YYYY.MM.DD): plugin.json, every SKILL.md
# frontmatter `version:` and every skill's "**Extension:** anofox_tabfm vYYYY.MM.DD" header line.
# Part of cutting a release; scripts/check_doc_snippets.py fails while they disagree or lag the
# latest v* tag.
#
#   scripts/bump_plugin_version.sh 2026.11.15
set -euo pipefail

version="${1:-}"
if [[ ! "$version" =~ ^[0-9]{4}\.[0-9]{2}\.[0-9]{2}$ ]]; then
    echo "usage: $0 YYYY.MM.DD" >&2
    exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
plugin="$root/plugins/anofox-tabfm"

sed -i.bak -E "s/(\"version\": \")[0-9]{4}\.[0-9]{2}\.[0-9]{2}(\")/\1${version}\2/" \
    "$plugin/.claude-plugin/plugin.json"
for skill in "$plugin"/skills/*/SKILL.md; do
    sed -i.bak -E \
        -e "s/^version: [0-9]{4}\.[0-9]{2}\.[0-9]{2}$/version: ${version}/" \
        -e "s/(\*\*Extension:\*\* anofox_tabfm v)[0-9]{4}\.[0-9]{2}\.[0-9]{2}/\1${version}/" \
        "$skill"
done
find "$plugin" -name '*.bak' -delete
python3 "$root/scripts/check_doc_snippets.py" --static
