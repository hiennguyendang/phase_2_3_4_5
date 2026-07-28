from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "11-calibrate_report.py"
# Phase-local scripts intentionally use flat imports; isolate the phase-3
# constants module when unittest discovers both phase test packages together.
sys.modules.pop("constants", None)
SPEC = importlib.util.spec_from_file_location("m3_report_calibration", SCRIPT)
CAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CAL)


def args():
    return SimpleNamespace(
        min_negatives=10, min_unique_patients=10, min_calls=10,
        target_ppv=0.90, target_specificity=0.90, target_npv=0.95,
        target_concept_ppv=0.90, target_concept_specificity=0.90,
        min_concept_positives=10, min_concept_f1=0.55, min_concept_auc=0.75,
        max_ci_width=None, bootstrap=0, seed=7,
    )


class CalibrationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.target = np.r_[np.ones(60, dtype=np.int8), np.zeros(60, dtype=np.int8)]
        grid_aligned_negative = np.linspace(0.01, 0.99, 99)[4]
        self.prob = np.r_[np.full(60, 0.90), np.full(60, grid_aligned_negative)]
        self.patient = np.asarray([f"p{i:08d}" for i in range(120)])

    def test_disease_dual_threshold_and_unknown_contract(self):
        item = CAL.fit_disease(self.prob, self.target, self.patient, args())
        self.assertTrue(item["present_supported"])
        self.assertTrue(item["absent_supported"])
        self.assertLess(item["absent_threshold"], item["present_threshold"])
        self.assertAlmostEqual(
            item["present_rate"] + item["absent_rate"] + item["unknown_rate"], 1.0)

    def test_unknown_labels_do_not_become_negative(self):
        target = np.full(120, -100, dtype=np.int8)
        item = CAL.fit_disease(self.prob, target, self.patient, args())
        self.assertEqual(item["n"], 0)
        self.assertFalse(item["supported"])
        self.assertIsNone(item["present_threshold"])

    def test_concept_gate_is_present_only(self):
        item = CAL.fit_concept(self.prob, self.target, self.patient, args())
        self.assertTrue(item["allowed_for_why"])
        self.assertIsNotNone(item["present_threshold"])
        self.assertNotIn("absent_threshold", item)

    def test_auc_uses_average_ranks_for_ties(self):
        scores = np.asarray([0.5, 0.5, 0.5, 0.5])
        target = np.asarray([1, 1, 0, 0], dtype=np.int8)
        self.assertAlmostEqual(CAL.auc_binary(scores, target), 0.5)


if __name__ == "__main__":
    unittest.main()
