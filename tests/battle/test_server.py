import socket

import pytest

from flymon.battle.server import NODE_BIN, ShowdownServer

pytestmark = pytest.mark.skipif(not NODE_BIN.exists(), reason="run scripts/install_showdown.sh first")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_write_config_is_loopback():
    srv = ShowdownServer(port=8123)
    text = srv.write_config()
    assert "exports.bindaddress = '127.0.0.1'" in text
    assert "exports.port = 8123" in text


def test_start_stop_and_health():
    port = _free_port()
    with ShowdownServer(port=port) as srv:
        assert srv.is_up()
        assert srv.url_ws == f"ws://127.0.0.1:{port}/showdown/websocket"
    assert not srv.is_up()


def test_pack_and_validate_team():
    srv = ShowdownServer(port=_free_port())
    packed = srv.pack_team("Blastoise\n- Surf\n- Earthquake\n- Strength\n- Reflect\n")
    assert packed.startswith("Blastoise||||Surf,Earthquake,Strength,Reflect")
    ok, msg = srv.validate_team(packed)
    assert ok, msg
    bad = srv.pack_team("Blastoise\n- Surf\n- Fire Blast\n")
    ok, msg = srv.validate_team(bad)
    assert not ok and "Fire Blast" in msg
