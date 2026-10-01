#!/usr/bin/env python3
"""Spec P.6.4 (the operating characteristic of P.6.2's whole rule, recorded before the judgement run; n stays 32
whatever it says).

Steps:
1. Refuse unless block o2 of the committed O summary is a judged declared run whose per-seed rows are in O2's cache
   (results/o/run/cache/o2_arm/, written under the block's measure key; D reproduced for both X), unless C3's z equals
   O2's, and unless block n1's dissimilar-pair o agrees with -2.433 within 5e-4 (c1).
2. Map the directions onto O2's rows: r1 (X = 4:1) <- O2's X = 4:1 (its Y was 1:4: the stated limitation), r2 (X = δ-DL)
   <- O2's X = δ-DL (Y = 4:1, the same pair).
3. p_rules.oc_run: oc_draws experiments of 32 seeds resampled from O2's, each judged by P.6.2's rule with the real
   bootstrap, under the observed effects and under c1 nulls (both directions; each direction alone; the null shifts only
   s by mean(ℓ) − c1 with t fixed, p_cli.OC_NULL, stated in the block). No pool is built. Block oc is a record: it
   changes no label and not n.

    uv run python scripts/p_oc.py                                          # the controller only (before R1)
    uv run python scripts/p_oc.py --smoke --allow-dirty

Exit 0 when recorded, 2 for a refusal."""
from __future__ import annotations

import json
import sys

from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.p_cli import (c1_source, check_committed, code_keys, git_state, hooks, load_c3,  # noqa: F401
                                m0d_sha, main_stage, make_measurer, model_types, o2_source, spec_commit)
from flymon.brain.p_rules import o2_vectors, oc_run, oc_sentence

LIMITATION = ("P.6.4: r1(X = 4:1, Y = δ-DL)은 처벌 조건화로 돈 적이 없어 O2의 X = 4:1 행(Y = 1:4)으로 대신했다 — "
              "r1의 운영 특성은 근사다. n은 32로 고정하며 이 결과로 바꾸지 않는다.")


def prepare(spec, smoke: bool, hk: dict, doc: dict, summary, z) -> tuple:
    c1, why = hk["c1_source"](spec, smoke)
    if why:
        return None, why
    src, why = hk["o2_source"](spec, smoke)
    if why:
        return None, why
    if json.loads(json.dumps(z)) != json.loads(json.dumps(src["z"])):
        return None, f"C3's z {z} differs from O2's {src['z']}: O2's rows are not in P's units"
    return dict(c1=c1, o2=src), None


def body(ctx) -> tuple:
    spec, ex = ctx.spec, ctx.extra
    vec = o2_vectors(ex["o2"]["rows"], ex["o2"]["z"], spec, O_SPEC)
    res = oc_run(vec, ex["c1"]["c1"], spec)
    res.update(c1_source=ex["c1"], limitation=LIMITATION, null=ex["o2"]["null"],   # p_cli.OC_NULL
               o2_source={k: ex["o2"][k] for k in ("run_id", "measure_key", "D", "summary", "cache", "n_rows", "seeds",
                                                   "null")})
    return dict(res, sentence=oc_sentence(res)), 0


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("oc", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__, prepare=prepare)


if __name__ == "__main__":
    sys.exit(main())
