# tests/brain/test_q_runner.py
"""The Q stage chain (Readings 3, 15, 19): repro -> smoke -> s_c -> q0 -> q1 (six conditions, one per call) ->
records; each stage refuses (exit 2, nothing written) when an earlier block is missing, a later one exists, the summary
is uncommitted or a hashed file is dirty; a failed reproduction gate blocks smoke; the mv fallback per side; a gated
condition outside the KC band is INVALID and recorded, not refused. (c) per Q.6.9: smoke picks the lowered s 0.7 / 0.8
by the KC band, both out -> s None ("(c) 조작 불가"), q1:s_up INVALID without running and ① 판단 불가."""
import hashlib
import json
from pathlib import Path

import pytest

from flymon.brain import q_runner as QR
from flymon.brain.config import Params
from flymon.brain.q_spec import SPEC
from tests.brain.q_fixtures import KEYS, READOUT, TYPES, Z, fake_result, fake_rows

SHA = {"base": "sha-base", "s_up": "sha-base", "apl_mbon05": "sha-m05", "apl_nonkc": "sha-nonkc", "mv_lo": "sha-lo",
       "mv_hi": "sha-hi"}
EDGES = {"apl_mbon05": 2, "apl_nonkc": 394}
S0, S1 = SPEC.s_down


class Scripted:
    def __init__(self, q0, kc_for=lambda p: 0.05, kc_cond=None, alter_repro=False, kc_s=None):
        self.q0, self.kc_for, self.kc_cond, self.alter = q0, kc_for, kc_cond or {}, alter_repro
        self.kc_s = kc_s
        self.calls, self.probes, self.last_wall_s, self.last_jobs = [], [], 1.5, 0

    def run(self, rows, cond, block, seeds):
        self.calls.append((cond.name, block, len(rows)))
        out = []
        for r in rows:
            k = QR.key_str(r)
            i = KEYS.index(k)
            n, n_act = len(seeds["report"]), len(seeds["act"])
            if block == "repro":
                res = json.loads(json.dumps(self.q0[k]))
                if self.alter:
                    res["alpha_reward"] = 0.2 if res["alpha_reward"] != 0.2 else 0.5
                res["q"] = fake_result(n, n_act, 20, -6, 2.0, i)["q"]
            else:
                res = fake_result(n, n_act, px=4 + 3 * (i % 7), dpx=-(1.0 + i % 5), sd=2.0, seed=1000 + i,
                                  sha=SHA[cond.name], edges=EDGES.get(cond.name, 0),
                                  kc=self.kc_cond.get(cond.name, 0.05))
            out.append(dict(key=k, result=res, cache_key=f"ck{i}", cache_file=f"results/q/cache/oracle/{i}.json"))
        return out

    def kc_probe(self, odours, params, strength, seeds):
        self.probes.append((params.mv_per_synapse, strength))
        v = self.kc_s(strength) if self.kc_s and strength != SPEC.strength else self.kc_for(params)
        return {oid: [v] * len(seeds) for oid in odours}


def _world(tmp_path, monkeypatch, pairs_r=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(QR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(QR, "git_state", lambda: dict(commit="c", dirty_hashed=[], dirty_other=[]))
    rows = fake_rows()
    q0, man = {}, []
    d = tmp_path / SPEC.q0_cache_dir
    d.mkdir(parents=True)
    for i, r in enumerate(rows):
        k = QR.key_str(r)
        res = fake_result(8, 8, px=20, dpx=(-12 if i >= 12 else -0.5), sd=(1.0 if i >= 12 else 3.0), seed=i, q=False)
        q0[k] = res
        f = d / f"{i:024d}.json"
        f.write_text(json.dumps({"kind": "oracle", "result": res}))
        man.append(dict(axis="b", turn=r["turn"], x=r["x"], y=r["y"], cache_key=f"key{i}",
                        cache_file=f"results/encoder/cache/oracle/{i:024d}.json",
                        cache_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    cfg = {"s": 1.0, "manifest": man}
    if pairs_r is not None:
        cfg["pairs"] = pairs_r(rows)
    enc = {"even": {"code_key": "c", "configs": {"k2-norm": cfg}}}
    ctx = dict(rows=rows, enc=enc, readout=READOUT, z=Z, types=TYPES, params=Params(), n_kc=100,
               max_rate_hz=200.0, cap_hz=1000.0 / 3.0, encoder_keys=lambda sel: [True] * len(sel))
    return ctx, q0


@pytest.fixture
def world(tmp_path, monkeypatch):
    return _world(tmp_path, monkeypatch)


def _runner(ctx, m):
    return QR.Runner(m, ctx, SPEC, code={"key": "q" * 64})


def _doc():
    return json.loads(Path(SPEC.summary).read_text())


def test_full_chain(world):
    ctx, q0 = world
    m = Scripted(q0)
    r = _runner(ctx, m)
    assert r.stage_repro()["passed"]
    sm = r.stage_smoke()
    assert sm["mv"] == [0.8, 1.25]
    assert sm["edit_edges"]["apl_mbon05"] == [SPEC.apl_mbon05_edges] and sm["edit_edges"]["apl_nonkc"] == [394]
    assert sm["conditions"]["apl_mbon05"]["status"] == "OK"
    assert sm["s_cap_record"]["s"] == 1.3                # the cap-limited max s: a record only (vmax 1.25 here)
    sc = r.stage_s_c()
    assert sc["s"] == S0 and sc["weak"] is False and sc["direction"] == "lowered" and not sc["unmanipulable"]
    assert sc["s_cap_record"]["s"] == 1.3 and "하향" in sc["note"]
    assert r.stage_q0()["status"] == "OK"
    for n in SPEC.cond_names:
        assert r.stage_q1(n)["status"] == "OK", n
    assert _doc()["q1:s_up"]["condition"]["strength"] == S0 and "하향" in _doc()["q1:s_up"]["note"]
    out = r.stage_records()
    doc = _doc()
    assert set(doc) == set(QR.ORDER)                     # canonical JSON sorts keys; the chain order is _require's
    assert set(out["candidates"]) == {"variation", "floor", "apl", "reach", "operating"}
    assert doc["records"]["fs"]["disagree"] >= 0 and len(doc["records"]["sentences"]) == 5
    assert ("base", "q1", 21) in m.calls and ("base", "repro", 3) in m.calls
    assert ("s_up", "smoke", 2) in m.calls
    assert all(c[2] == 2 for c in m.calls if c[1] == "smoke")
    assert (Params().mv_per_synapse, S0) in m.probes and not any(p[1] == S1 for p in m.probes)


def test_s_c_falls_back_to_second_lowered_s(world):
    ctx, q0 = world
    m = Scripted(q0, kc_s=lambda s: 0.2 if s == S0 else 0.05)
    r = _runner(ctx, m)
    r.stage_repro()
    sm = r.stage_smoke()
    assert sm["s_choice"]["s"] == S1 and set(sm["s_choice"]["kc_probe_median"]) == {str(S0), str(S1)}
    sc = r.stage_s_c()
    assert sc["s"] == S1 and sc["weak"] is False and sc["kc_probe_median"][str(S0)] == 0.2
    r.stage_q0()
    for n in SPEC.cond_names:
        r.stage_q1(n)
    assert _doc()["q1:s_up"]["condition"]["strength"] == S1


def test_s_c_both_lowered_out_of_band_is_unmanipulable(world):
    ctx, q0 = world
    m = Scripted(q0, kc_s=lambda s: 0.2)
    r = _runner(ctx, m)
    r.stage_repro()
    sm = r.stage_smoke()
    assert sm["s_choice"]["s"] is None and sm["s_choice"]["unmanipulable"]
    assert "s_up" not in sm["conditions"] and not any(c[0] == "s_up" for c in m.calls)
    sc = r.stage_s_c()
    assert sc["s"] is None and sc["weak"] is False and sc["unmanipulable"] and sc["unmanip_note"] == SPEC.unmanip_note
    r.stage_q0()
    for n in SPEC.cond_names:
        r.stage_q1(n)
    blk = _doc()["q1:s_up"]
    assert blk["status"] == "INVALID" and "(c) 조작 불가" in blk["reasons"][0] and blk["condition"]["strength"] is None
    assert not any(c[0] == "s_up" for c in m.calls)       # never run
    assert all(_doc()[f"q1:{n}"]["status"] == "OK" for n in SPEC.cond_names if n != "s_up")
    out = r.stage_records()
    fl = out["candidates"]["floor"]
    assert fl["label"] == "판단 불가" and SPEC.unmanip_note in fl["why"]


def test_order_and_rewrite_refusals(world):
    ctx, q0 = world
    r = _runner(ctx, Scripted(q0))
    with pytest.raises(SystemExit) as e:
        r.stage_q0()
    assert e.value.code == 2 and not Path(SPEC.summary).exists()
    r.stage_repro()
    r.stage_smoke()
    with pytest.raises(SystemExit):
        r.stage_repro()                                  # a later block exists
    with pytest.raises(SystemExit):
        r.stage_q1("base")                               # s_c and q0 missing


def test_repro_failure_blocks_smoke(world):
    ctx, q0 = world
    r = _runner(ctx, Scripted(q0, alter_repro=True))
    out = r.stage_repro()
    assert out["passed"] is False and not all(p["equal"] for p in _doc()["repro"]["pairs"])
    with pytest.raises(SystemExit):
        r.stage_smoke()


def test_repro_fails_on_encoder_key_mismatch(world):
    ctx, q0 = world
    ctx = dict(ctx, encoder_keys=lambda sel: [True, False, True])
    assert _runner(ctx, Scripted(q0)).stage_repro()["passed"] is False


def test_mv_fallback_per_side(world):
    ctx, q0 = world
    base_mv = Params().mv_per_synapse
    kc = lambda p: 0.02 if p.mv_per_synapse < base_mv else 0.05  # noqa: E731
    r = _runner(ctx, Scripted(q0, kc_for=kc))
    r.stage_repro()
    sm = r.stage_smoke()
    assert sm["mv"] == [0.9, 1.25] and sm["mv_fallback_used"] == [True, False]


def test_kc_band_invalidates_a_gated_condition_and_is_recorded(world):
    ctx, q0 = world
    r = _runner(ctx, Scripted(q0, kc_cond={"s_up": 0.2, "base": 0.2}))
    r.stage_repro()
    r.stage_smoke()
    r.stage_s_c()
    r.stage_q0()
    assert r.stage_q1("base")["status"] == "OK"
    for n in SPEC.cond_names[1:]:
        r.stage_q1(n)
    assert _doc()["q1:s_up"]["status"] == "INVALID"
    out = r.stage_records()
    assert out["candidates"]["floor"]["label"] in ("불일치", "판단 불가")


def _pairs_r(delta=0.0):
    def make(rows):
        from flymon.brain import q_records
        out = []
        for i, r in enumerate(rows):
            v = q_records.pair_values(fake_result(8, 8, px=20, dpx=(-12 if i >= 12 else -0.5),
                                                  sd=(1.0 if i >= 12 else 3.0), seed=i, q=False), Z, SPEC)
            out.append(dict(axis="b", turn=r["turn"], x=r["x"], y=r["y"], r=v["r"] + (delta if i == 3 else 0.0)))
        out.append(dict(axis="a", turn=0, x="p", y="q", r=9.9))
        return out
    return make


def _to_q0(r):
    r.stage_repro()
    r.stage_smoke()
    r.stage_s_c()
    return r.stage_q0()


def test_q0_encoder_r_check_match(tmp_path, monkeypatch):
    ctx, q0 = _world(tmp_path, monkeypatch, pairs_r=_pairs_r())
    chk = _to_q0(_runner(ctx, Scripted(q0)))["encoder_r_check"]
    assert chk["available"] and chk["encoder_grid_r_match"] is True and chk["max_abs_diff"] == 0.0
    assert chk["n_compared"] == SPEC.n_b and chk["missing"] == []


def test_q0_encoder_r_check_mismatch(tmp_path, monkeypatch):
    ctx, q0 = _world(tmp_path, monkeypatch, pairs_r=_pairs_r(0.25))
    chk = _to_q0(_runner(ctx, Scripted(q0)))["encoder_r_check"]
    assert chk["encoder_grid_r_match"] is False and chk["max_abs_diff"] == pytest.approx(0.25)
    assert chk["diff"][KEYS[3]] == pytest.approx(-0.25)


def test_q0_encoder_r_check_unavailable(world):
    ctx, q0 = world
    chk = _to_q0(_runner(ctx, Scripted(q0)))["encoder_r_check"]
    assert chk["available"] is False and chk["encoder_grid_r_match"] is None


def test_dirty_summary_or_hashed_files_refuse(world, monkeypatch):
    ctx, q0 = world
    monkeypatch.setattr(QR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    r = _runner(ctx, Scripted(q0))
    Path(SPEC.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(SPEC.summary).write_text("{}")
    with pytest.raises(SystemExit):
        r.stage_repro()
    monkeypatch.setattr(QR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(QR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/q_jobs.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_repro()


def test_script_has_main_guard_no_allow_dirty_and_refuses_wrong_cwd(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_q.py").read_text()
    assert 'if __name__ == "__main__":' in text
    assert 'add_argument("--allow-dirty"' not in text
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_q", root / "scripts/run_q.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    assert mod.main(["--stage", "s_c"]) == 2
