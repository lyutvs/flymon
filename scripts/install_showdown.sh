#!/usr/bin/env bash
# Install the pinned Pokémon Showdown server for FlyMon (loopback only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)/flymon/battle"
cd "$ROOT"
npm ci --no-audit --no-fund
mkdir -p node_modules/pokemon-showdown/logs/repl
echo "installed pokemon-showdown $(node -e "console.log(require('./node_modules/pokemon-showdown/package.json').version)")"
