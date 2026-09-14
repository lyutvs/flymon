"""Build data/malecns.npz from the three MaleCNS v1.0 feather files (CC-BY 4.0, HHMI Janelia/Google).

Usage: uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from .connectome import SIGN_OF_NT, Connectome

WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOT = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = "body-neurotransmitters-male-cns-v1.0.feather"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _side(ann) -> np.ndarray:
    """Sensory somas sit outside the volume (somaSide 'M'); use rootSide when it is L/R."""
    soma = ann["somaSide"].fillna("").astype(str)
    root = ann["rootSide"].fillna("").astype(str)
    side = root.where(root.isin(["L", "R"]), soma)
    return side.where(side.isin(["L", "R"]), "M").to_numpy().astype(str)


def build(data_dir: str | Path, out_path: str | Path, min_weight: int = 1) -> dict:
    data_dir, out_path = Path(data_dir), Path(out_path)
    ann = feather.read_table(
        data_dir / ANNOT,
        columns=["bodyId", "type", "class", "superclass", "somaSide", "rootSide", "status"],
    ).to_pandas()
    ann = ann[(ann["status"] == "Traced") & ann["type"].notna()].sort_values("bodyId").reset_index(drop=True)
    ids = ann["bodyId"].to_numpy(dtype=np.int64)

    nt = feather.read_table(data_dir / NT, columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    ntv = nt.reindex(ids).fillna("unknown").to_numpy().astype(str)
    sign = np.array([SIGN_OF_NT.get(x, 0) for x in ntv], dtype=np.int8)

    # streaming edge read: the weights file is ~1.1 GB, so go batch by batch
    lookup = np.full(ids.max() + 1, -1, dtype=np.int64) if ids.max() < 50_000_000 else None
    if lookup is not None:
        lookup[ids] = np.arange(len(ids))

    def to_index(b: np.ndarray) -> np.ndarray:
        if lookup is not None:
            ok = (b >= 0) & (b < len(lookup))
            out = np.full(b.shape, -1, np.int64)
            out[ok] = lookup[b[ok]]
            return out
        i = np.searchsorted(ids, b)
        i[i >= len(ids)] = 0
        return np.where(ids[i] == b, i, -1)

    P, Q, W = [], [], []
    reader = pa.ipc.open_file(pa.memory_map(str(data_dir / WEIGHTS)))
    for bi in range(reader.num_record_batches):
        bt = reader.get_batch(bi)
        w = bt.column("weight").to_numpy(zero_copy_only=False).astype(np.int64)
        k = w >= min_weight
        if not k.any():
            continue
        pre = to_index(bt.column("body_pre").to_numpy(zero_copy_only=False)[k].astype(np.int64))
        post = to_index(bt.column("body_post").to_numpy(zero_copy_only=False)[k].astype(np.int64))
        ok = (pre >= 0) & (post >= 0)
        P.append(pre[ok].astype(np.int32)); Q.append(post[ok].astype(np.int32)); W.append(w[k][ok].astype(np.int32))
    pre = np.concatenate(P) if P else np.zeros(0, np.int32)
    post = np.concatenate(Q) if Q else np.zeros(0, np.int32)
    w = np.concatenate(W) if W else np.zeros(0, np.int32)

    conn = Connectome(
        bodyId=ids,
        type=ann["type"].to_numpy().astype(str),
        cls=ann["class"].fillna("").to_numpy().astype(str),
        sc=ann["superclass"].fillna("").to_numpy().astype(str),
        nt=ntv, sign=sign, side=_side(ann), pre=pre, post=post, w=w,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn.save(out_path)

    types = conn.type
    manifest = {
        "inputs": {f: {"sha256": _sha256(data_dir / f), "bytes": (data_dir / f).stat().st_size} for f in (WEIGHTS, ANNOT, NT)},
        "n_neurons": conn.N, "n_edges": conn.E, "min_weight": min_weight,
        "counts": {
            "Kenyon_Cell": int((conn.cls == "Kenyon_Cell").sum()),
            "MBON": int((conn.cls == "MBON").sum()),
            "DAN": int((conn.cls == "DAN").sum()),
            "ALPN": int((conn.cls == "ALPN").sum()),
            "olfactory": int((conn.cls == "olfactory").sum()),
            "APL": int((types == "APL").sum()),
            "lLN": int(np.char.startswith(types, "lLN1").sum() + np.char.startswith(types, "lLN2").sum()),
        },
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="data/malecns.npz")
    ap.add_argument("--min-weight", type=int, default=1)
    a = ap.parse_args()
    m = build(a.data, a.out, a.min_weight)
    print(json.dumps({k: m[k] for k in ("n_neurons", "n_edges", "counts")}, indent=2))


if __name__ == "__main__":
    main()
