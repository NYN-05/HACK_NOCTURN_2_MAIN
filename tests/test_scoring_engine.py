from __future__ import annotations

import unittest

from engine.scoring_engine import ScoringEngine


class ScoringEngineTests(unittest.TestCase):
    def test_unavailable_layers_do_not_participate_in_fusion(self) -> None:
        engine = ScoringEngine(weights={"cnn": 0.7, "vit": 0.3})
        breakdown = engine.fuse(
            {"cnn": 92.0, "vit": 8.0},
            reliabilities={"cnn": 1.0, "vit": 0.1},
            availability={"cnn": True, "vit": False},
        )

        self.assertEqual(breakdown.available_layers, ["cnn"])
        self.assertFalse(breakdown.abstained)
        self.assertEqual(breakdown.layer_scores, {"cnn": 92})
        self.assertGreaterEqual(breakdown.weighted_score, 90)
        self.assertAlmostEqual(breakdown.confidence, 0.7, places=4)

    def test_abstains_when_no_layers_are_available(self) -> None:
        engine = ScoringEngine(weights={"cnn": 0.7, "vit": 0.3})
        breakdown = engine.fuse({}, reliabilities={}, availability={"cnn": False, "vit": False})

        self.assertTrue(breakdown.abstained)
        self.assertEqual(breakdown.available_layers, [])
        self.assertEqual(breakdown.layer_scores, {})
        self.assertEqual(breakdown.confidence, 0.0)
        self.assertEqual(breakdown.weighted_score, 50)


if __name__ == "__main__":
    unittest.main()
