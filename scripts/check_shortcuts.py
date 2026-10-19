import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from countworld.scene import SceneCfg  # noqa: E402
from countworld.shortcuts import probe  # noqa: E402

cfg = SceneCfg(max_count=int(sys.argv[1]) if len(sys.argv) > 1 else 9)
for beta in (0.0, 1.5):
    for kind in ("nocount", "scene1", "scene2", "both"):
        print(f"beta={beta} {kind:8s} acc={probe(cfg, kind, beta=beta):.3f}", flush=True)
