#!/usr/bin/env python
"""Serve the live viewer page on 127.0.0.1 until Ctrl-C.

    uv run python scripts/live_viewer.py --port 8765
"""
from __future__ import annotations

import argparse
import sys
import time

from flymon.live.server import ViewerServer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--keep-battles", type=int, default=3, help="recent battles kept per fly for late pages")
    args = ap.parse_args(argv)
    with ViewerServer(port=args.port, keep_battles=args.keep_battles) as srv:
        print(f"viewer: {srv.url}/  (live_battles.py prints the URL with ?fly= and &showdown=)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
