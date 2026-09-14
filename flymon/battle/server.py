"""Local Pokémon Showdown server (pinned npm package), loopback only."""
from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

PKG_ROOT = Path(__file__).parent
NODE_BIN = PKG_ROOT / "node_modules" / "pokemon-showdown" / "pokemon-showdown"
CONFIG_DIR = PKG_ROOT / "node_modules" / "pokemon-showdown" / "config"
LOGS_REPL = PKG_ROOT / "node_modules" / "pokemon-showdown" / "logs" / "repl"

CONFIG_TEMPLATE = """'use strict';
exports.port = {port};
exports.bindaddress = '127.0.0.1';
exports.workers = 1;
exports.ssl = null;
exports.proxyip = false;
exports.loginserver = 'https://play.pokemonshowdown.com/';
exports.serverid = 'flymon';
exports.servertoken = '';
exports.repl = false;
exports.crashguard = true;
exports.reportjoins = false;
exports.noipchecks = true;
exports.emergency = false;
exports.potd = '';
exports.allowrequestingties = true;
exports.forcetimer = false;
exports.simulatorprocesses = 1;
exports.validatorprocesses = 1;
"""


class ShowdownServer:
    def __init__(self, root: Path = PKG_ROOT, port: int = 8000):
        self.root, self.port = Path(root), int(port)
        self.proc: subprocess.Popen | None = None

    @property
    def url_ws(self) -> str:
        return f"ws://127.0.0.1:{self.port}/showdown/websocket"

    @property
    def url_http(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def install(self) -> None:
        subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=self.root, check=True)
        LOGS_REPL.mkdir(parents=True, exist_ok=True)

    def write_config(self) -> str:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        text = CONFIG_TEMPLATE.format(port=self.port)
        (CONFIG_DIR / "config.js").write_text(text)
        return text

    def is_up(self) -> bool:
        try:
            with urllib.request.urlopen(self.url_http, timeout=1) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self, timeout_s: float = 30.0) -> "ShowdownServer":
        if not NODE_BIN.exists():
            raise FileNotFoundError(f"{NODE_BIN} missing: run scripts/install_showdown.sh")
        LOGS_REPL.mkdir(parents=True, exist_ok=True)
        self.write_config()
        self.proc = subprocess.Popen(
            ["node", str(NODE_BIN), "start", "--no-security", "--port", str(self.port)],
            cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.is_up():
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"showdown server exited with {self.proc.returncode}")
            time.sleep(0.25)
        self.stop()
        raise TimeoutError("showdown server did not come up")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- team tooling (no server needed) --------------------------------------------
    def pack_team(self, export_text: str) -> str:
        out = subprocess.run(["node", str(NODE_BIN), "pack-team"], input=export_text, text=True,
                             capture_output=True, check=True)
        return out.stdout.strip()

    def validate_team(self, packed: str, format_id: str = "gen1ou") -> tuple[bool, str]:
        out = subprocess.run(["node", str(NODE_BIN), "validate-team", format_id], input=packed + "\n", text=True,
                             capture_output=True)
        return out.returncode == 0, (out.stdout + out.stderr).strip()
