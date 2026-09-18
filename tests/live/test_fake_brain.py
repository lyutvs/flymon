"""The fake brain walks the brain's viewer path: same choice as its inner provider, detail + one trace per candidate."""
import numpy as np

from flymon.live.events import to_json
from flymon.live.fake_brain import FakeBrainProvider
from flymon.live.sink import RecordingSink


class Fixed:
    async def decide(self, battle, candidates, context):
        return 1


async def test_keeps_the_choice_and_reports_detail_and_traces():
    sink = RecordingSink()
    brain = FakeBrainProvider(Fixed(), sink, "fm-live-f00", seed=0)
    ctx = {"battle_tag": "battle-gen1ou-5", "turn": 3}
    assert await brain.decide(None, ["surf", "earthquake", "blizzard"], ctx) == 1
    d = ctx["detail"]
    assert set(d) == {"V", "p", "kc_active_frac"} and all(len(v) == 3 for v in d.values())
    assert int(np.argmax(d["V"])) == 1 and abs(float(np.sum(d["p"])) - 1.0) < 0.01
    to_json(d)                                                   # numpy detail serialises
    assert [(e["type"], e["slot"], e["phase"], e["turn"]) for e in sink.events] == \
        [("trace", s, "decide", 3) for s in range(3)]
    paths = {s["path"]: s for s in sink.events[0]["series"]}
    assert set(paths) == {"events", "rate_hz/alpn", "rate_hz/kc", "rate_hz/mbon", "rate_hz/apl", "cell_hz/mbon"}
    assert len(paths["rate_hz/kc"]["points"]) == 14 and paths["cell_hz/mbon"]["kind"] == "bars"
    starts = [e["series"][0]["points"][0][0] for e in sink.events]
    assert starts == sorted(starts) and starts[1] > starts[0]    # one running clock, like SpikeTap
