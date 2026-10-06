"""AA.8: the four STOP labels and their verbatim sentences with the candidate counts; the result sentence (no verdict
words, k = 1 variant); reuse / screen-machine reasons; the budget arithmetic of AA.7 2 / 5; env reasons."""
from flymon.brain import aa_rules as R
from flymon.brain.aa_spec import SPEC
from flymon.brain.y_spec import SPEC as Y

C = dict(untouched=218, screened=31, trained=0)


def test_labels_are_four():
    assert R.STOPS == ("STOP_REUSE", "STOP_BUDGET", "STOP_MACHINE", "STOP_NO_PAIRS")


def test_every_stop_sentence_ends_with_the_candidate_counts():
    for body in (R.reuse_stop(["x"], C), R.budget_stop("1.0 h + 12.0 h = 13.0 h", "2", "주 세트 미측정", C),
                 R.machine_stop("순진 거름", "단위 수", "학습 측정 없음, 31쌍 `screened`", C),
                 R.no_pairs_stop(dict(balance=20, a_floor=8, p_floor=3), C)):
        assert body["sentence"].endswith(R.tail(C)) and body["outcome"] in R.STOPS
    assert R.tail(C) == "(후보 상태: untouched 218 · screened 31 · trained 0)"


def test_no_pairs_sentence_is_verbatim():
    s = R.no_pairs_stop(dict(balance=20, a_floor=8, p_floor=3), C)["sentence"]
    assert s.startswith("AA 순진 거름(판정 시드 순진 pre, F 8 × K 8)에서 주 세트 오라클 관대 통과 31쌍 가운데 Y 최종 거름"
                        "(|d′| < 0.5 ∧ MBON13(X) ≥ 20 ∧ MBON05(X) ≥ 43)을 통과한 쌍이 0이다(관대 → 엄격 재현율 0/31; "
                        "탈락 사유별 개수 균형 20 · MBON13(X) 바닥 8 · MBON05(X) 바닥 3)")
    assert "31쌍 `screened`, `trained` 0, 218쌍 `untouched` — 사용자 몫." in s


def test_reason_counts_use_y_rounding_and_overlap():
    rows = [dict(d=0.5, L_A=30.0, L_P=60.0), dict(d=0.1, L_A=19.9999999999, L_P=60.0), dict(d=0.9, L_A=10.0, L_P=10.0)]
    # row 1: |0.5| is not < 0.5; row 2: 19.9999999999 rounds (9 digits) to 20.0 and passes; row 3 fails all three
    assert R.reason_counts(rows, Y) == dict(balance=2, a_floor=1, p_floor=1)


def test_result_sentence_has_no_verdict_word_and_switches_at_k1():
    base = dict(k=12, g=6, few=False, mu_ra=0.8, lo_ra=0.2, hi_ra=1.4, mu_pa=-0.6, lo_pa=-1.1, hi_pa=-0.1,
                method="two_stage", cov=0.93, under=False, tau_ra=0.4, tau_pa=0.3, phi_r=0.05, phi_p=0.2, rng_min=None, rng_max=None)
    s = R.result_sentence(base)
    assert "0.8 [95% CI 0.2, 1.4]" in s and "-0.6 [-1.1, -0.1]" in s and "묶음-마리 두 단계" in s
    assert "쌍 간 SD 보상 0.4 · 처벌 0.3(DL)" in s                     # plan Reading 25
    assert "판정 문턱은 없고 M2 학습 단위의 PASS/FAIL이 아니다." in s
    assert not any(w in s.replace("PASS/FAIL이 아니다", "") for w in R.FORBIDDEN_RESULT_WORDS)
    few = R.result_sentence(dict(base, k=3, g=2, few=True, method="fly", rng_min=0.1, rng_max=1.3, under=True))
    assert "'적은 묶음' 표식" in few and "마리 단계(묶음 < 5, 쌍별 값 범위 0.1–1.3)" in few and "'명목 미달'" in few
    one = R.result_sentence(dict(base, k=1, key="b|9|x|y"))
    assert one.startswith("쌍별 연구(k = 1) — 쌍 b|9|x|y의 쌍별 추정") and "통합" not in one


def test_budget_arithmetic():
    assert R.core_ok(1.0, (12.0 - 1.0) / 1.3, SPEC) and not R.core_ok(1.0, 8.47, SPEC)
    assert R.spent_ok(12.0, SPEC) and not R.spent_ok(12.0000001, SPEC)


def test_screen_machine_reasons():
    units = [dict(pair="p", fly=f, brain="naive", probe_seeds=[62_000_000 + f * 100 + k for k in range(8)])
             for f in range(8)]
    want = {("p", f): u["probe_seeds"] for f, u in enumerate(units)}
    counts = {("p", f): [[[1, 2], [3, 4]]] * 8 for f in range(8)}
    assert R.screen_machine_reasons(units, counts, want, n_units=8, k=8) == []
    assert R.screen_machine_reasons(units + units[:1], counts, want, n_units=8, k=8)          # duplicate + count
    bad = dict(counts)
    bad[("p", 0)] = [[[1, 2], [3, 4]]] * 7
    assert any("K 8" in r for r in R.screen_machine_reasons(units, bad, want, n_units=8, k=8))
    seeds = dict(want)
    seeds[("p", 1)] = [1] * 8
    assert any("시드" in r for r in R.screen_machine_reasons(units, counts, seeds, n_units=8, k=8))


def test_env_reasons_added_file_is_not_a_difference():
    a = dict(numpy="2", aa_files={"a.py": "1"})
    assert R.env_reasons([("stage0", a)], dict(numpy="2", aa_files={"a.py": "1", "b.py": "2"})) == []
    assert R.env_reasons([("stage0", a)], dict(numpy="3", aa_files={"a.py": "1"})) == ["stage0 대비 numpy"]


def _good_facts():
    ydoc = {b: dict(outcome=o) for b, o in SPEC.y_outcomes}
    ydoc["digest"]["set"] = dict(digest_keys=Y.main_digest_keys, n_b=Y.main_n_b, n_a=Y.main_n_a,
                                 last_turn=Y.main_last_turn)
    ydoc["oracle"].update(detail_sha256=SPEC.oracle_sha256, derived=dict(yield_rule=dict(n_len=SPEC.n_len)))
    return dict(y_anc={c: True for _, c in SPEC.y_blocks}, z_anc={c: True for _, c in SPEC.z_blocks}, y_doc=ydoc,
                z_doc={b: dict(outcome=o) for b, o in SPEC.z_outcomes},
                keys=dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key), lenient=None,
                axes=dict(SPEC.n_len_axes), decl_sha={"flymon/brain/y_spec.py": "s"},
                now_sha={"flymon/brain/y_spec.py": "s"}, absent={p: True for p in SPEC.y_absent_files})


def test_reuse_reasons_each_item_breaks_alone():
    assert R.reuse_reasons(_good_facts(), SPEC, Y) == []
    f = _good_facts()
    f["z_anc"][SPEC.z_blocks[0][1]] = False
    assert any("HEAD 이력" in w for w in R.reuse_reasons(f, SPEC, Y))
    f = _good_facts()
    f["y_doc"]["learn"] = dict(outcome="PASS")
    assert R.reuse_reasons(f, SPEC, Y) == ["Y learn 블록이 있음"]
    f = _good_facts()
    f["lenient"] = "sha mismatch"
    assert R.reuse_reasons(f, SPEC, Y) == ["oracle.json: sha mismatch"]
    f = _good_facts()
    f["axes"] = dict(b=10, a=21)
    assert len(R.reuse_reasons(f, SPEC, Y)) == 1
    f = _good_facts()
    f["now_sha"]["flymon/brain/y_spec.py"] = "t"
    assert len(R.reuse_reasons(f, SPEC, Y)) == 1
    f = _good_facts()
    f["absent"][SPEC.y_absent_files[0]] = False
    assert len(R.reuse_reasons(f, SPEC, Y)) == 1
    f = _good_facts()
    f["y_doc"]["digest"]["set"]["n_b"] = 1
    f["keys"]["u_measure_key"] = "x"
    assert len(R.reuse_reasons(f, SPEC, Y)) == 2
