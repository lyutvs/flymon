"""AD.6 5: before the analysis, every log / checkpoint / wall_clock.json under results/m4/ad/ and the run load log are
listed with their sha256 (committed with the result) and packed into a tar.gz outside git; verify() names any file that
changed or vanished, and an archive unpacked into results/m4/ad/ verifies."""
import tarfile

from flymon.ad import raw


def tree(root):
    (root / "logs").mkdir(parents=True)
    (root / "logs/fly00.jsonl").write_text('{"a": 1}\n')
    (root / "checkpoints/learn").mkdir(parents=True)
    (root / "checkpoints/learn/state_000001.npz").write_bytes(b"\x00\x01")
    (root / "wall_clock.json").write_text("{}")


def test_listing_archive_and_verify(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "results/m4/ad"
    tree(root)
    log = tmp_path / "load.log"
    log.write_text("x\n")
    entries = raw.listing("results/m4/ad", extra=["load.log"])
    assert [e["path"] for e in entries] == ["load.log", "results/m4/ad/checkpoints/learn/state_000001.npz",
                                           "results/m4/ad/logs/fly00.jsonl", "results/m4/ad/wall_clock.json"]
    arc = raw.archive(entries, tmp_path / "arc/ad-raw.tar.gz")
    doc = dict(files=entries, archive=arc)
    assert raw.verify(doc) == []
    (root / "logs/fly00.jsonl").write_text('{"a": 2}\n')
    (root / "wall_clock.json").unlink()
    assert raw.verify(doc) == ["results/m4/ad/logs/fly00.jsonl", "results/m4/ad/wall_clock.json"]
    with tarfile.open(arc["path"]) as t:                 # recovery: unpack the archive, then verify the list
        t.extractall(tmp_path, filter="data")
    assert raw.verify(doc) == []
