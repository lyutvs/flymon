"""J.10 pre-amendment dry run (no engine): read gain_probe.json (C3 engine, all51 glomeruli at ORN gains 0.25-16),
interpolate log response vs log gain per glomerulus, and run two gain-update rules for three updates toward the
geometric mean of the g = 1 responses. Targets: KC spikes and ALPN spikes. Gains are clamped to the probed range, so
residual spread at 0.25 is partly a probe-range artefact (strong glomeruli need stronger attenuation)."""
import collections
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/j/diag/gain_probe.json"))
rows, gains = d["rows"], np.array(d["gains"])
lg = np.log(gains)
agg = collections.defaultdict(list)
for r in rows:
    for k in ("kc", "pn", "orn_hz", "cmd_hz"):
        agg[(r["g_type"], r["gain"], k)].append(r[k])
types = sorted({r["g_type"] for r in rows})
mean = lambda t, g, k: float(np.mean(agg[(t, g, k)]))
rank = lambda a: np.argsort(np.argsort(a))

kc1 = np.array([mean(t, 1.0, "kc") for t in types]); pn1 = np.array([mean(t, 1.0, "pn") for t in types])
print(f"g=1: KC spread {kc1.max() / kc1.min():.1f}x, ALPN spread {pn1.max() / pn1.min():.1f}x, "
      f"spearman(ALPN, KC) {np.corrcoef(rank(pn1), rank(kc1))[0, 1]:.2f}")
for g in gains:
    print(f"g={g}: cmd {np.median([mean(t, g, 'cmd_hz') for t in types]):.0f} Hz, realized median "
          f"{np.median([mean(t, g, 'orn_hz') for t in types]):.0f} max {max(mean(t, g, 'orn_hz') for t in types):.0f} Hz")
ks = [np.log(mean(t, 2.0, "kc") / mean(t, 1.0, "kc")) / np.log(2) for t in types if mean(t, 1.0, "kc") > 5]
print(f"KC exponent k(g 1->2): median {np.median(ks):.2f}, IQR {np.percentile(ks, 25):.2f}-{np.percentile(ks, 75):.2f}")


def f(t, key, g):
    y = np.log(np.maximum([mean(t, gg, key) for gg in gains], 0.5))
    return float(np.exp(np.interp(np.log(g), lg, y)))


for key in ("kc", "pn"):
    y1 = np.array([f(t, key, 1.0) for t in types]); G = np.exp(np.log(y1).mean())
    print(f"== target {key}: G {G:.1f}, spread {y1.max() / y1.min():.1f}x, in [G/2, 2G] {int(((y1 >= G / 2) & (y1 <= 2 * G)).sum())}/51")
    for name in ("proportional", "log-secant"):
        g, prev = np.ones(len(types)), None
        for _ in range(3):
            y = np.array([f(t, key, gi) for t, gi in zip(types, g)])
            if name == "proportional" or prev is None:
                gn = g * G / y
            else:
                pg, py = prev
                dx = np.log(g) - np.log(pg)
                s = np.clip((np.log(y) - np.log(py)) / np.where(np.abs(dx) > 1e-9, dx, 1.0), 0.2, 5.0)
                gn = np.exp(np.log(g) + (np.log(G) - np.log(y)) / s)
            prev, g = (g, y), np.clip(gn, gains.min(), gains.max())
        y = np.array([f(t, key, gi) for t, gi in zip(types, g)])
        print(f"  {name}: after 3 updates spread {y.max() / y.min():.2f}x, in [G/2, 2G] "
              f"{int(((y >= G / 2) & (y <= 2 * G)).sum())}/51, g {g.min():.2f}-{g.max():.2f}, at probe bounds "
              f"{int(((g <= gains.min()) | (g >= gains.max())).sum())}")
