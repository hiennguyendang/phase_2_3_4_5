from __future__ import annotations

import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Avoid reusing phase-3's flat ``constants`` module during combined discovery.
sys.modules.pop("constants", None)
from assemble import concept_evidence, disease_thresholds  # noqa: E402
from run import _validate_m3_provenance  # noqa: E402


class StrictReadoutTests(unittest.TestCase):
    def setUp(self):
        self.region = "left lung"
        self.disease = "Lung Opacity"
        self.concept = "lung opacity"
        self.record = {
            "regions": {
                self.region: {
                    "disease_concepts": {
                        self.disease: {
                            self.concept: {"prob": 0.91, "edge_weight": 2.0,
                                           "contribution": 1.82},
                            "fracture": {"prob": 0.99, "edge_weight": 9.0,
                                         "contribution": 8.91},
                        }
                    }
                }
            }
        }
        self.gate = {
            "region_by_name": {
                self.region: {
                    self.concept: {"allowed_for_why": True, "present_threshold": 0.80},
                    "fracture": {"allowed_for_why": True, "present_threshold": 0.80},
                }
            }
        }

    def test_no_gate_means_no_visible_evidence(self):
        self.assertEqual(concept_evidence(self.record, self.region, self.disease, None), [])

    def test_pair_gate_and_graph_mask_are_both_required(self):
        evidence = concept_evidence(
            self.record, self.region, self.disease, concept_gate=self.gate)
        self.assertEqual([x["concept"] for x in evidence], [self.concept])
        self.assertEqual(evidence[0]["contribution"], 1.82)

    def test_unsupported_pair_does_not_fallback(self):
        thresholds = {"region_by_name": {self.region: {self.disease: {
            "present_threshold": None, "absent_threshold": None, "supported": False,
        }}}}
        self.assertEqual(disease_thresholds(
            self.disease, thresholds, "region", self.region), (None, None))

    def test_provenance_rejects_missing_record_identity(self):
        artifact = {"_artifact": {"provenance": {
            "checkpoint_sha256": "checkpoint-a",
            "manifest_sha256": "manifest-a",
            "box_source": "detector",
        }}}
        with self.assertRaises(SystemExit):
            _validate_m3_provenance({"image": {"box_source": "detector"}}, artifact, "test")

    def test_provenance_accepts_exact_identity(self):
        artifact = {"_artifact": {"provenance": {
            "checkpoint_sha256": "checkpoint-a",
            "manifest_sha256": "manifest-a",
            "box_source": "detector",
        }}}
        record = {"m3_checkpoint_sha256": "checkpoint-a",
                  "m3_manifest_sha256": "manifest-a", "box_source": "detector"}
        _validate_m3_provenance({"image": record}, artifact, "test")


if __name__ == "__main__":
    unittest.main()
