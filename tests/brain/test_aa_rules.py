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
                method="two_stage", cov=0.93, under=False, tau_ra=0.4, tau_pa=0.3, phi_r=0.05, phi_p=0.2, rng_ra=None, rng_pa=None)
    s = R.result_sentence(base)
    assert "0.8 [95% CI 0.2, 1.4]" in s and "-0.6 [-1.1, -0.1]" in s and "묶음-마리 두 단계" in s
    assert "쌍 간 SD 보상 0.4 · 처벌 0.3(DL)" in s                     # plan Reading 25
    assert "판정 문턱은 없고 M2 학습 단위의 PASS/FAIL이 아니다." in s
    assert not any(w in s.replace("PASS/FAIL이 아니다", "") for w in R.FORBIDDEN_RESULT_WORDS)
    few = R.result_sentence(dict(base, k=3, g=2, few=True, method="fly", rng_ra=(0.1, 1.3), rng_pa=(-0.9, 0.2),
                                 under=True))
    assert "'적은 묶음' 표식" in few and "마리 단계(묶음 < 5, 쌍별 값 범위 보상 0.1–1.3 · 처벌 -0.9–0.2)" in few
    assert "'명목 미달'" in few
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
    f = _good_facts()
    f["y_doc"]["oracle"]["outcome"] = "STOP"
    assert R.reuse_reasons(f, SPEC, Y) == ["Y 블록 oracle의 결과 STOP ≠ PASS"]
    f = _good_facts()
    f["z_doc"]["sens"]["outcome"] = "PASS"
    assert R.reuse_reasons(f, SPEC, Y) == ["Z 블록 sens의 결과 PASS ≠ STOP_PLAN_UNREACHABLE"]
    f = _good_facts()
    f["y_doc"]["oracle"]["derived"]["yield_rule"]["n_len"] = SPEC.n_len + 1
    assert R.reuse_reasons(f, SPEC, Y) == [f"Y oracle 블록 N_len ≠ {SPEC.n_len}"]
    f = _good_facts()
    f["now_sha"]["flymon/brain/x_new.py"] = "n"                 # in now, missing from decl
    w = R.reuse_reasons(f, SPEC, Y)
    assert len(w) == 1 and "flymon/brain/x_new.py" in w[0]


# --- full-string oracles, typed from AA.8 (spec 6274–6278) by hand, not built from the module ---------------------
TAIL = " (후보 상태: untouched 218 · screened 31 · trained 0)"


def test_reuse_stop_full_string():
    s = R.reuse_stop(["a", "b"], C)
    assert s["sentence"] == ("AA 재사용 조건(AA.7 1)이 깨졌다(a; b). AA는 Y · Z 블록, 주 세트 오라클 상세(`oracle.json`), "
                             "주 세트 생성원, W 측정 경로를 다시 재거나 고치는 경로를 갖지 않으므로 주 세트 학습 측정 없이 "
                             "멈춘다 — 사용자 몫." + TAIL)
    assert s["outcome"] == "STOP_REUSE" and s["reasons"] == ["a", "b"] and s["candidates_count"] == C


def test_budget_stop_full_string():
    s = R.budget_stop("1.00 h + 11.50 h = 12.50 h", "5 전", "주 세트 미측정", C)
    assert s["sentence"] == ("남은 추정 비용 1.00 h + 11.50 h = 12.50 h가 AA 핵심 상한 12 h를 넘는다(5 전) — 주 세트 미측정."
                             + TAIL)
    assert s["outcome"] == "STOP_BUDGET"


def test_machine_stop_full_string():
    s = R.machine_stop("순진 거름", "단위 수 247 ≠ 248", "학습 측정 없음, 31쌍 `screened`", C)
    assert s["sentence"] == ("AA 순진 거름 측정에서 기계 검사가 맞지 않았다(단위 수 247 ≠ 248) — 같은 쌍으로 다시 돌리지 "
                             "않으며(W.5 · W.9.8 H8), `INVALID_RUN` 여부는 사용자 몫이다. 학습 측정 없음, 31쌍 `screened`."
                             + TAIL)
    assert s["outcome"] == "STOP_MACHINE"


def test_no_pairs_stop_full_string():
    s = R.no_pairs_stop(dict(balance=20, a_floor=8, p_floor=3), C)
    assert s["sentence"] == (
        "AA 순진 거름(판정 시드 순진 pre, F 8 × K 8)에서 주 세트 오라클 관대 통과 31쌍 가운데 Y 최종 거름(|d′| < 0.5 ∧ "
        "MBON13(X) ≥ 20 ∧ MBON05(X) ≥ 43)을 통과한 쌍이 0이다(관대 → 엄격 재현율 0/31; 탈락 사유별 개수 균형 20 · "
        "MBON13(X) 바닥 8 · MBON05(X) 바닥 3) — 1차 집합이 비어 학습 측정이 1차 추정에 아무것도 보태지 않으므로 주 세트 "
        "학습 측정 없이 멈춘다. 31쌍 `screened`, `trained` 0, 218쌍 `untouched` — 사용자 몫." + TAIL)
    assert s["outcome"] == "STOP_NO_PAIRS" and s["reasons"] == ["k = 0"]


_RES = dict(k=12, g=6, few=False, mu_ra=0.8, lo_ra=0.2, hi_ra=1.4, mu_pa=-0.6, lo_pa=-1.1, hi_pa=-0.1,
            method="two_stage", cov=0.93, under=False, tau_ra=0.4, tau_pa=0.3, phi_r=0.05, phi_p=0.2,
            rng_ra=None, rng_pa=None)
_HEAD = ("조합 지렛대 모델(C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, z_V)에서 "
         "실제 학습 규칙(F.2 R · N + G.5 RN)의 ")
_ASSOC = ("보상 연합 d′(ΔV_R1 − ΔV_N1)은 0.8 [95% CI 0.2, 1.4], 처벌 연합 d′((ΔV_R2 − ΔV_R1) − (ΔV_RN2 − ΔV_RN1))는 "
          "-0.6 [-1.1, -0.1]로 추정됐다")
_FILT = "판정 시드 순진 거름(|d′| < 0.5 ∧ MBON13(X) ≥ 20 ∧ MBON05(X) ≥ 43)을 통과한"


def _floor(r, p):
    return (f"가르친 세포 바닥 몫 중앙 φ_R {r} · φ_P {p} — 바닥 절단을 포함한 값이며(편향 부호 미정) Y 거름 통과(같은 "
            "프로브 실현값에서) 쌍 조건부다. 판정 문턱은 없고 M2 학습 단위의 PASS/FAIL이 아니다.")


def test_result_sentence_full_string_two_stage():
    # "쌍 간 SD 보상 〈τ̂_RA〉 · 처벌 〈τ̂_PA〉(DL)" = plan Reading 25's split of the spec's single 〈τ̂〉
    want = (_HEAD + _ASSOC + " — 주 세트 오라클 관대 통과 31쌍 중 " + _FILT + " 12쌍(X 냄새 묶음 6개), F 8 × K 8, 쌍 추정 = "
            "마리 d′(±10 잘라냄)의 마리 평균, 통합 = 쌍 평균 · 묶음-마리 두 단계 부트스트랩 10 000회 백분위(구조 맞춤 포함 "
            "확률 0.93). 쌍 간 SD 보상 0.4 · 처벌 0.3(DL). " + _floor("0.05", "0.2"))
    assert R.result_sentence(_RES) == want


def test_result_sentence_full_string_fly_few_under():
    # "쌍별 값 범위 보상 〈min_RA〉–〈max_RA〉 · 처벌 〈min_PA〉–〈max_PA〉" = plan Reading 27 (per association, no envelope)
    r = dict(_RES, k=3, g=2, few=True, under=True, method="fly", rng_ra=(0.1, 1.3), rng_pa=(-1.25, -0.05),
             phi_r=None)
    want = (_HEAD + _ASSOC + " — 주 세트 오라클 관대 통과 31쌍 중 " + _FILT + " 3쌍(X 냄새 묶음 2개, '적은 묶음' 표식), F 8 × "
            "K 8, 쌍 추정 = 마리 d′(±10 잘라냄)의 마리 평균, 통합 = 쌍 평균 · 마리 단계(묶음 < 5, 쌍별 값 범위 보상 0.1–1.3 · 처벌 "
            "-1.25–-0.05) "
            "부트스트랩 10 000회 백분위(구조 맞춤 포함 확률 0.93, '명목 미달'). 쌍 간 SD 보상 0.4 · 처벌 0.3(DL). "
            + _floor("null", "0.2"))
    assert R.result_sentence(r) == want


def test_result_sentence_full_string_k1():
    one = R.result_sentence(dict(_RES, k=1, key="b|9|x|y"))
    want = ("쌍별 연구(k = 1) — 쌍 b|9|x|y의 쌍별 추정: " + _HEAD + _ASSOC + " — 주 세트 오라클 관대 통과 31쌍 중 " + _FILT
            + " 1쌍, F 8 × K 8, 쌍 추정 = 마리 d′(±10 잘라냄)의 마리 평균. " + _floor("0.05", "0.2"))
    assert one == want
    assert "X 냄새 묶음" not in one and "쌍 간 SD" not in one


def test_number_format_has_no_negative_zero():
    s = R.result_sentence(dict(_RES, mu_pa=-0.0001, lo_pa=-0.0004, hi_pa=0.0002))
    assert "는 0 [0, 0]로 추정됐다" in s and "-0" not in s.replace("-0.6", "")


def test_screen_machine_reasons_nan_count():
    units = [dict(pair="p", fly=f, probe_seeds=[f * 10 + k for k in range(8)]) for f in range(2)]
    want = {("p", f): u["probe_seeds"] for f, u in enumerate(units)}
    counts = {("p", f): [[[1, 2], [3, 4]]] * 8 for f in range(2)}
    counts[("p", 1)] = [[[1, 2], [3, float("nan")]]] * 8
    assert R.screen_machine_reasons(units, counts, want, n_units=2, k=8) == ["p fly 1: 카운트가 유한하지 않음"]


def test_reuse_reasons_empty_decl_sha_fails_closed():
    f = _good_facts()
    f["decl_sha"], f["now_sha"] = {}, {}
    w = R.reuse_reasons(f, SPEC, Y)
    assert len(w) == 1 and "비어 있음" in w[0]
