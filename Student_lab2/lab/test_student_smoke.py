"""Starter self-checks: add tests after implementing your pipeline."""
import unittest
from pathlib import Path

from feature_pipeline import (
    load_scada,
    load_las,
    merge_inputs,
    engineer_features,
)

DATA_DIR = Path(__file__).parent / "data"
class FeaturePipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scada = load_scada(DATA_DIR / "scada_synthetic.csv")
        cls.las = load_las(DATA_DIR / "well_b_synthetic.las")
        cls.merged = merge_inputs(cls.las, cls.scada)
        cls.features = engineer_features(cls.merged)

    def test_merge_returns_nine_matched_rows(self):
        self.assertEqual(len(self.features), 9)
        self.assertEqual(
            self.features[["well_id", "depth_m"]].duplicated().sum(),
            0,
        )

    def test_missing_sensor_values_are_preserved(self):
        self.assertEqual(
            self.features["pressure_psi"].isna().sum(), 1
        )
        self.assertEqual(
            self.features["flow_m3_h"].isna().sum(), 1
        )

        pressure_row = self.features[
            self.features["depth_m"] == 1005
        ].iloc[0]

        self.assertEqual(pressure_row["pressure_missing_flag"], 1)

        flow_row = self.features[
            self.features["depth_m"] == 1007
        ].iloc[0]

        self.assertEqual(flow_row["flow_missing_flag"], 1)

    def test_feature_calculations(self):
        row = self.features[
            self.features["depth_m"] == 1001
        ].iloc[0]

        self.assertAlmostEqual(row["pressure_delta_psi"], 44.0)
        self.assertAlmostEqual(row["temperature_delta_c"], 1.8)

        variance_row = self.features[
            self.features["depth_m"] == 1002
        ].iloc[0]

        self.assertAlmostEqual(
            variance_row["pressure_var_3"],
            1290.6666667,
            places=5,
        )

if __name__ == "__main__":
    unittest.main()

