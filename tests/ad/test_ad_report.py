"""AD.3's sentence frames and the assembled result: the primary sentence with the chance-switch null, the crossed CI,
the reference row, the valid flies and later drop-outs, the KC tail and the label; the summary by primary 1 with the
40-battle note and the final-point reason; STOP_FUTILE; the learning-check sentence; notes on C-off / descriptive rows;
the forbidden-word check."""
import pytest

from flymon.ad import report
from flymon.ad.spec import LABEL

C = dict(name="1차 1", a="FLYL", b="FLYOS", point=120, n_a=12, n_b=11, min_a=9, min_b=9, status="충족", diff=0.2,
         lo=0.05, hi=0.35, crossed=dict(lo=-0.01, hi=0.41))
REF = dict(name="기준 행", a="FLYL", b="FLYOS", point=0, status="기술", diff=-0.03, lo=-0.1, hi=0.04,
           crossed=dict(lo=-0.15, hi=0.09))


def test_primary_sentence():
    s = report.primary_sentence(C, REF, 0.183, 1, 0.31, notes=("예산 소진 — 최종 판독 시점 120",))
    assert s.startswith("FLY-L − FLY-OS(우연 전환 대조, 120배틀 시점)에서 4.3 기준 1: 0.200 [95% CI 0.050, 0.350] — 탐색 기준 충족.")
    assert "FLY-OS 쪽은 우연 전환이다(자기 잡음 기대치 Σq·q 0.183)." in s
    assert "확인 세트 20쌍에 한함(교차 부트스트랩 95% CI -0.010, 0.410, 기술)." in s
    assert "기준 행 FLY-L(0) − FLY-OS(0): -0.030 [95% CI -0.100, 0.040]." in s
    assert "유효 마리 FLY-L 12 · FLY-OS 11(120배틀 시점, 이후 탈락 1)." in s and "예산 소진 — 최종 판독 시점 120" in s
    assert s.endswith(f"{LABEL} 같은 턴 후보 KC 비율 > 2 턴 비율(FLY-L): 0.310.")
    u = report.primary_sentence(dict(C, status="측정 불성립", n_b=8), REF, None, 0, None)
    assert "유효 마리 FLY-L 12/9 · FLY-OS 8/9 — 탐색 기준 측정 불성립." in u and "기록 없음" in u


def test_row_sentences_and_notes():
    c = dict(name="FLY-L(40) − C-off", a="FLYL", b="COFF", point=40, n_a=12, n_b=6, min_a=9, min_b=4, status="기술",
             diff=0.1, lo=0.0, hi=0.2, crossed=dict(lo=-0.05, hi=0.25))
    s = report.row_sentence(c)
    assert report.C_OFF_NOTE in s and report.MULTI_NOTE in s and s.endswith(LABEL)
    assert report.unmeasured_sentence(4) == f"4.3 기준 4: — 탐색 기준 미측정(AC.2 2단계 팔 보류, 부록 AD). {LABEL}"


def test_summary_sentence():
    p2 = dict(C, point=40, status="미충족")
    s = report.summary_sentence(C, p2, 120, "complete")
    assert s.startswith("상대 타입을 실제 상대와 무관하게 섞고 총 세기를 같게 둔 우연 전환 대조(FLY-OS)와 비교해, 배틀로 120배틀")
    assert "확인 세트 20쌍은 1단계에서 이미 쓰였고, 그 상황은 학습 중에도 나왔을 수 있다(AC.4 · AD.0 3)." in s
    assert "40배틀 시점 대조는 미충족로 최종 시점과 다르다 — 둘을 합쳐 읽지 않는다." in s and s.endswith(LABEL)
    s = report.summary_sentence(dict(C, status="미충족"), dict(p2, status="미충족"), 80, "fallback")
    assert s.startswith("배틀로 80배틀 학습한 마리의 상성 조건부 선택이 상대 타입을 섞은 우연 전환 대조(FLY-OS)와 비교해 탐색 기준에 못 미쳤다")
    assert "최종 판독 시점은 80배틀이다(대체 80)." in s and "합쳐 읽지" not in s
    s = report.summary_sentence(dict(C, status="측정 불성립"), p2, 60, "STOP_BUDGET")
    assert s.startswith("AD 1차 대조는 측정 불성립이다") and "(예산 소진의 마지막 공통 시점)" in s


def test_futility_and_learncheck_sentences():
    f = dict(status="STOP_FUTILE", share_at_point=0.31, share_at_0=0.35, point=40)
    assert report.futility_sentence(f) == ("STOP_FUTILE: 40배틀 시점 FLY-L 맞는 방향 쌍 비율 0.310가 0 시점 0.350보다 크지 "
                                           f"않아 학습을 40배틀에서 멈췄다(판정 아님). 판독은 40배틀 시점이다. {LABEL}")
    lcd = dict(right_share={"FLYL": {0: dict(share=0.2), 20: dict(share=0.3), 40: dict(share=0.4)}},
               switch_curve={"FLYL": {0: dict(mean=0.1), 40: dict(mean=0.2)}}, null={"FLYOS": {40: 0.19}})
    s = report.learncheck_sentence("FLYL", lcd, (0, 20, 40), 40)
    assert s == ("FLY-L 학습 확인(기록, 판정 아님): 맞는 방향 쌍 비율 0 0.200 → 20 0.300 → 40 0.400; 전환율 0 0.100 → 40 0.200; "
                 f"FLY-OS 잡음 기대치 Σq·q 40 0.190. {LABEL}")


def test_check_refuses_forbidden_words():
    report.check(["탐색 기준 충족."])
    with pytest.raises(ValueError):
        report.check(["FLY PASS"])
    with pytest.raises(ValueError):
        report.check(["학습 안 됨"])


# ---- report.analyse on synthetic records (carry notes: S2, D4, Task 8 stopped / point=None, Task 17 floor row) ----
import dataclasses  # noqa: E402
import inspect  # noqa: E402

from flymon.ad import analysis as an  # noqa: E402
from flymon.ad.spec import SPEC  # noqa: E402

S = dataclasses.replace(SPEC, boot_draws=200, learn_battles=40)
PAIRS = [dict(cands=["m0", "m1"], best1="m0", best2="m1", mults1=[2.0, 0.5], mults2=[0.5, 2.0]) for _ in range(2)]
MI = {"m0": ("WATER", 90), "m1": ("FIRE", 40)}


def srec(g, point, pos, floor=0.0):
    sits = []
    for i, p in enumerate(pos):
        sits.append(dict(pair=i, side=0, v=[1.0, 0.0] if p else [0.0, 1.0], pick=0 if p else 1))
        sits.append(dict(pair=i, side=1, v=[0.0, 1.0] if p else [1.0, 0.0], pick=1 if p else 0))
    return dict(fly=g, point=point, arm="FLYL" if g < 36 else "FLYOS", set="confirm", mode="own", frozen=True,
                situations=sits, switched=[int(p) for p in pos], floor_contact=floor, weights=dict(frac_changed=0.1))


def make_sits(l_pts, os_pts, floor_l=None):
    floor_l = floor_l or {}
    out = []
    for g in range(24, 36):
        out += [srec(g, p, [p > 0, (p + g) % 3 == 0], floor_l.get(p, 0.0)) for p in l_pts]
    for g in range(36, 48):
        out += [srec(g, p, [(p + g) % 2 == 0, 0]) for p in os_pts]
    return out


def run(sits, status="COMPLETE", stopped=None, futility=None, monkeypatch=None):
    result = dict(status=status, futility=futility, books={"L": dict(invalid={}, stopped=stopped or {})},
                  per_fly=None)
    if monkeypatch is not None:
        monkeypatch.setattr(an, "eval_readings", lambda *a, **k: [dict(
            criterion=3, name="평가 배틀 기준 3(승률) FLY-L − FLY-OS", a="FLYL", b="FLYOS", point=None, n_a=12,
            n_b=12, min_a=9, min_b=9, status="기술", diff=0.1, lo=-0.1, hi=0.3)])
    return report.analyse(spec=S, T=40, result=result, sits=sits, pairs=PAIRS, coff=[], fly1=[], battles_L=[],
                          eval_records=[], learn_logs={}, nobrain_rows=[], move_info=MI, kc_by_arm={"FLYL": 0.2},
                          load_summary=None)


def test_analyse_drops_unused_params():                                   # ruling D4
    ps = inspect.signature(report.analyse).parameters
    assert "new_pairs" not in ps and "all_records" not in ps


def test_floor_row_and_criteria_2_3_floor_note(monkeypatch):              # Task 17 carry + ruling S2
    sits = make_sits((0, 20, 40), (0, 20, 40), floor_l={20: 0.3, 40: 0.7})
    out = run(sits, monkeypatch=monkeypatch)
    fr = out["learncheck"]["floor_reading"]
    assert (fr["first_over"], fr["last_ok"]) == (40, 20)
    row = out["floor_row"]
    assert row["point"] == 20 and row["status"] in ("충족", "미충족") and "crossed" in row   # rule=True
    note = "회복 r 보정 범위(1 000펄스) 밖 — 바닥 접촉 0.700"
    assert note in out["sentences"][0]                                    # primary 1 at 40
    assert any("병기 행" in s and s.startswith("FLY-L − FLY-OS(우연 전환 대조, 20배틀 시점)") for s in out["sentences"])
    c23 = [s for s in out["sentences"] if s.startswith("평가 배틀 기준 3")]
    assert len(c23) == 1 and note in c23[0]
    assert all(s.endswith(LABEL) or "KC 비율" in s for s in out["sentences"])


def test_no_floor_note_below_read(monkeypatch):
    out = run(make_sits((0, 20, 40), (0, 20, 40), floor_l={20: 0.1, 40: 0.2}), monkeypatch=monkeypatch)
    assert out["floor_row"] is None and not any("회복 r" in s for s in out["sentences"])


def test_stopped_arm_final_point_and_dropouts():                          # Task 8 carry
    sits = make_sits((0, 20, 40), (0, 20))                                # FLY-OS stopped after 20, no invalid marks
    out = run(sits, status="STOP_BUDGET", stopped={"FLYOS": "STOP_INFRA: test"})
    assert out["final_point"] == dict(point=40, reason="STOP_BUDGET")     # the stopped arm is left out
    d = [x for x in out["dropouts"] if x["arm"] == "FLYOS"]
    assert len(d) == 12 and all(x["marked"] is False and x["last_point"] == 20 for x in d)
    assert out["primary"][0]["status"] == "측정 불성립"
    assert "예산 소진 — 최종 판독 시점 40" in out["sentences"][0]


def test_all_arms_stopped_point_none():                                   # Task 8 carry: point=None
    sits = make_sits((0, 20), (0, 20))
    out = run(sits, status="STOP_BUDGET", stopped={"FLYL": "STOP_INFRA: a", "FLYOS": "STOP_INFRA: b"})
    assert out["final_point"]["point"] is None
    assert out["primary"][0]["status"] == "측정 불성립" and out["primary"][0]["point"] is None
    assert out["summary"].startswith("AD 1차 대조는 측정 불성립이다(최종 판독 시점 없음") and "None" not in out["summary"]
    assert out["futility_and_learncheck_sentences"] == [] and len(out["dropouts"]) == 24


def test_futility_recheck_only_with_every_point40_record():               # Task 8 carry
    fut = dict(status="CONTINUE", point=40, share_at_point=None, share_at_0=None, n_at_point=12, n_at_0=12)
    sits = make_sits((0, 20, 40), (0, 20))                                # FLY-OS has no point-40 records
    out = run(sits, futility=fut)
    assert out["futility_recheck"]["checked"] is False
    assert run(make_sits((0, 20), (0, 20)), status="STOP_BUDGET")["futility_recheck"] is None   # no decision taken
    sits = make_sits((0, 20, 40), (0, 20, 40))
    from flymon.ad import validity
    good = validity.futility(sits, PAIRS, S)
    assert run(sits, futility=good)["futility_recheck"] == dict(checked=True, same=True)
    with pytest.raises(SystemExit, match="STOP_INFRA"):
        run(sits, futility=dict(good, share_at_point=0.0))


def test_summary_carries_the_floor_note(monkeypatch):                    # fix round 1, I1 (AD.3)
    note = "회복 r 보정 범위(1 000펄스) 밖 — 바닥 접촉 0.700"
    out = run(make_sits((0, 20, 40), (0, 20, 40), floor_l={20: 0.3, 40: 0.7}), monkeypatch=monkeypatch)
    assert note in out["summary"] and out["summary"].endswith(LABEL)
    out = run(make_sits((0, 20, 40), (0, 20, 40), floor_l={20: 0.1, 40: 0.2}), monkeypatch=monkeypatch)
    assert "회복 r" not in out["summary"]
    assert report.summary_sentence(C, C, 120, "complete", ["X"]).endswith(f"X {LABEL}")


def test_futility_recheck_reads_the_run_shape():                          # fix round 1, (a)
    import json

    from flymon.ad import validity
    sits = make_sits((0, 20, 40), (0, 20, 40))
    # run_ad.decide_futility returns validity.futility's dict (via persist_futility); run_ad writes it to result.json
    fut = json.loads(json.dumps(validity.futility(sits, PAIRS, S)))
    assert set(fut) >= {"status", "point", "share_at_point", "share_at_0", "n_at_point", "n_at_0"}
    assert run(sits, futility=fut)["futility_recheck"] == dict(checked=True, same=True)
    assert run(sits, futility={"status": "CONTINUE"})["futility_recheck"]["checked"] is True   # runner fallback shape


def test_every_situation_row_has_a_point():                               # fix round 1, (b)
    sits = make_sits((0, 20, 40), (0, 20, 40))
    extra = [dict(r, set="new") for r in sits if r["point"] in (0, 40)]
    extra += [dict(r, mode="os_paired") for r in sits if r["arm"] == "FLYL"]
    rd = an.situation_readings(sits + extra, S, final=40, new_points=[0, 40], coff=[], fly1=[], retried=[])
    rows = [c for k in ("did", "aux", "new_set", "paired_os", "sensitivity") for c in rd[k]]
    assert rows and all(isinstance(c["point"], int) for c in rows)
    assert {k for k in ("did", "aux", "new_set", "paired_os", "sensitivity") if rd[k]} == {
        "did", "aux", "new_set", "paired_os", "sensitivity"}
