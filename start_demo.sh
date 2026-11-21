#!/usr/bin/env bash
# One-click start for the digital twin demo (Linux/macOS). Options: see scripts/start_demo.py --help
cd "$(dirname "$0")" && exec python3 scripts/start_demo.py "$@"
