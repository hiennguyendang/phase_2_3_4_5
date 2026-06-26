"""Self-contained M5 demo — no model needed. Builds a few synthetic M3/M4 rows (same schema as
phase_3/infer.py and phase_4/infer.py) and runs the assembler, so the tier-1..6 logic can be
exercised before real predictions exist.

    python phase_5/demo.py
"""

from __future__ import annotations

import json

from assemble import assemble_image, realize_template
from run import run
from verify import verify

M3 = {
    "demo_with_prior": {
        "image_id": "demo_with_prior",
        "image_disease": {"Atelectasis": 0.83, "Pleural Effusion": 0.30, "Pneumonia": 0.05},
        "regions": {"left lower lung zone": {"disease": {"Atelectasis": 0.80}},
                    "right costophrenic angle": {"disease": {"Pleural Effusion": 0.55}}},
    },
    "demo_no_prior": {
        "image_id": "demo_no_prior",
        "image_disease": {"Cardiomegaly": 0.90},
        "regions": {"cardiac silhouette": {"disease": {"Cardiomegaly": 0.85}}},
    },
    "demo_normal": {"image_id": "demo_normal", "image_disease": {"Atelectasis": 0.04}, "regions": {}},
}
M4 = {
    "demo_with_prior": {
        "image_id": "demo_with_prior", "prior_image_id": "demo_prior",
        "regions": {"left lower lung zone": {"Atelectasis": {"prog": "worsened", "prob": 0.7,
                                                             "probs": [0.2, 0.1, 0.7]}}},
    },
}


def main() -> int:
    for iid, m3rec in M3.items():
        rep = assemble_image(m3rec, M4.get(iid))
        rep["text"] = realize_template(rep)
        rep["verify"] = verify(rep, rep["text"])
        print(f"\n=== {iid}  (has_prior={rep['has_prior']}, normal={rep['normal']}) ===")
        print("REPORT:", rep["text"])
        for f in rep["findings"]:
            print(f"  - {f['disease']} [{f['status']} {f['prob']}] @ {f['lead_region']} "
                  f"temporal={f['temporal']}  <- provenance {json.dumps(f['provenance'])}")
        print("VERIFY:", rep["verify"])

    print("\n--- corpus stats (template realizer) ---")
    _, stats = run(M3, M4, realize="template")
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
