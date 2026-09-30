"""Spec N.1 / N.8: every N number in one object; seed blocks new and disjoint; training seeds (~2.1e10) reach the
engine's generator unchanged; data pins equal the committed files; writes only under results/n/ and the N summary."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import n_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.h3_store import canonical, sha256_file
from flymon.brain.l_spec import SPEC as L_SPEC
from flymon.brain.n_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]


def _declared(spec):
    return [spec.act_seeds, spec.select_seeds, spec.report_seeds, spec.pilot_seeds, spec.judge_seeds(max(spec.n_grid)),
            spec.n0f_seeds]


def test_declared_sizes_and_grids():
    assert (len(SPEC.act_seeds), len(SPEC.select_seeds), len(SPEC.report_seeds), len(SPEC.pilot_seeds),
            len(SPEC.n0f_seeds)) == (16, 8, 16, 16, 8)
    assert (SPEC.act_seeds[0], SPEC.select_seeds[0], SPEC.report_seeds[0], SPEC.pilot_seeds[0], SPEC.judge_seed0,
            SPEC.n0f_seeds[0]) == (21_000_000, 21_000_100, 21_000_200, 21_001_000, 21_002_000, 21_003_000)
    assert SPEC.g_grid == (0.125, 0.25, 0.5, 1.0, 2.0, 4.0) and SPEC.c_delta_grid == (1.0, 2.0, 4.0, 8.0)
    assert SPEC.n_grid == (16, 24, 32, 48, 64) and SPEC.sd_mults == (1.0, 2.0)
    assert SPEC.judge_seeds(3) == (21_002_000, 21_002_001, 21_002_002)
    assert SPEC.mixture("4:1") == {"IA": 0.8, "EB": 0.2} and SPEC.mixture("1:4") == {"IA": 0.2, "EB": 0.8}
    assert SPEC.pair_stimuli() == {"sim": ("4:1", "1:4"), "dis": ("4:1", "dDL")}
    assert SPEC.edit_of("on") == "none" and SPEC.edit_of("block") == "apl_to_kc_zero"
    assert SPEC.edit_of("all") == "apl_all_zero"


def test_seed_blocks_are_new_and_disjoint_and_training_seeds_never_meet_them():
    flat = [s for b in _declared(SPEC) for s in b]
    assert len(flat) == len(set(flat))
    h4 = SPEC.h4
    old = set(h4.act_seeds) | set(h4.select_seeds) | set(h4.report_seeds) | set(h4.teach_seeds)
    assert not old & set(flat)
    train = [SPEC.train_seed(s, t) for s in flat for t in range(h4.teach_trials)]
    assert len(set(train)) == len(train) and min(train) > max(flat)


def test_training_seed_is_train_blocks_rule():
    assert SPEC.train_seed(21_002_063, 11) == 1_000_000 + 21_002_063 * 1000 + 11 == 21_003_063_011


def test_big_seeds_reach_the_engine_generator_unchanged(synthetic_connectome):
    c = synthetic_connectome()
    e = Engine(c, Populations.from_connectome(c), Params(min_weight=1, balance_hemispheres=False), seed=0)
    big = SPEC.train_seed(max(SPEC.judge_seeds(max(SPEC.n_grid))), SPEC.h4.teach_trials - 1)
    assert big > 2 ** 32
    e.reset(big)
    a = e.rng.random()
    assert a == np.random.default_rng(big).random()
    e.reset(big % 2 ** 32)
    assert e.rng.random() != a                                     # no silent 32-bit truncation
    assert json.loads(canonical({"s": big}))["s"] == big           # cache keys keep the exact integer


def test_constants_come_through_l_and_h4():
    assert SPEC.l is L_SPEC and SPEC.h4 is L_SPEC.j.h4 and SPEC.h3 is L_SPEC.j.h4.h3
    assert (SPEC.zero_share_max, SPEC.testable_min) == (0.25, 2.0)
    assert (SPEC.h4.teach_trials, SPEC.h4.teach_present_ms, SPEC.h4.teach_gap_ms) == (12, 800.0, 200.0)
    assert SPEC.h3.strength == 0.35 and SPEC.h3.punish_type == "PPL105"


def test_data_pins_are_the_committed_files():
    for name, sha in SPEC.sha_pins().items():
        assert sha256_file(ROOT / SPEC.data_dir / name) == sha
    text = (ROOT / "scripts" / "fetch_door_hallem.py").read_text()   # the pins' one home: loaded, not restated
    assert all(v in text for v in (SPEC.door_repo, SPEC.door_commit, SPEC.door_column, *SPEC.sha_pins().values()))
    assert SPEC.n_receptors == 24 and dict(SPEC.odorant_cas)["dDL"] == "705-86-2"
    assert SPEC.door_commit not in (ROOT / "flymon" / "brain" / "n_spec.py").read_text()


def test_smoke_is_small_and_outside_every_declared_block():
    s = smoke(SPEC)
    assert s.g_grid == (1.0,) and s.c_delta_grid == (1.0,) and s.n_grid == (3,)
    declared = {x for b in _declared(SPEC) for x in b}
    smoke_seeds = [x for b in _declared(s) for x in b]
    assert not declared & set(smoke_seeds) and len(smoke_seeds) == len(set(smoke_seeds))
    assert s.sha_pins() == SPEC.sha_pins()                          # smoke runs on the real, pinned data


def test_store_writes_only_under_results_n_and_the_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = Params()
    n_store.write_json("results/n/x/a.json", {"a": 1}, [p])
    assert json.loads(Path("results/n/x/a.json").read_text()) == {"a": 1}
    n_store.write_summary_block(n_store.SUMMARY, "n0f", {"b": 2}, [p])
    n_store.write_summary_block(n_store.SUMMARY, "n0", {"c": 3}, [p])
    assert json.loads(Path(n_store.SUMMARY).read_text()) == {"n0f": {"b": 2}, "n0": {"c": 3}}
    for bad in ("results/m0d/n/a.json", "results/summary/m_readout.json", "results/n_other.json"):
        with pytest.raises(SystemExit):
            n_store.write_json(bad, {}, [p])
