#!/usr/bin/env bash
# Запуск из папки проекта: ./run.sh   или   ./run.sh lab
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
# Без аргументов — графическое приложение с настройками; иначе передать в main (например lab, --quick)
if [ $# -eq 0 ]; then
  exec python3 -m vkr_terrain.desktop_app
else
  exec python3 -m vkr_terrain.main "$@"
fi
