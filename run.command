#!/usr/bin/env bash
# macOS: двойной щелчок в Finder — откроется Terminal и запустится демо.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
if [ $# -eq 0 ]; then
  python3 -m vkr_terrain.desktop_app
else
  python3 -m vkr_terrain.main "$@"
fi
echo
read -r -p "Enter — закрыть окно " _
