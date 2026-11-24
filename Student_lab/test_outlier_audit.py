
"""Week 11 Class 2 — Unit tests for Wine Chemistry Outlier Audit."""

import unittest
from pathlib import Path
import pandas as pd
from outlier_audit import audit_data



class TestOutlierAudit(unittest.TestCase):

    def setUp(self):
        """Create sample data for testing."""

        self.data = pd.DataFrame({
            "sample_id": ["W001", "W002", "W003", "W004", "W005"],
            "alcohol": [13.0, 12.5, 13.2, 12.8, 13.1],
            "malic_acid": [2.0, 2.2, 2.1, 2.3, 2.0],
            "color_intensity": [5.0, 4.8, 5.2, 5.1, 4.9],
            "proline": [800, 850, 820, 810, 830],
            "target": [0, 0, 1, 1, 2]
        })

    def test_missing_columns(self):
        """Missing required columns should raise ValueError."""

        incomplete_data = self.data.drop(columns=["proline"])

        with self.assertRaises(ValueError):
            audit_data(incomplete_data)

    def test_zero_variance_columns(self):
        """Zero variance should not cause division errors."""

        constant_data = self.data.copy()

        constant_data["malic_acid"] = 2.0
        constant_data["proline"] = 800

        results = audit_data(constant_data)

        self.assertFalse(
            results["retained"]["malic_acid_zscore_flag"].any()
        )

        self.assertFalse(
            results["retained"]["proline_zscore_flag"].any()
        )

    def test_invalid_records_are_quarantined(self):
        """Invalid domain values must be quarantined."""

        test_data = self.data.copy()

        test_data.loc[0, "malic_acid"] = -1
        test_data.loc[1, "proline"] = 0

        results = audit_data(test_data)

        self.assertEqual(len(results["quarantine"]), 2)
        self.assertEqual(len(results["retained"]), 3)

    def test_statistical_outliers_are_not_automatically_removed(self):
        """A statistical outlier alone should remain retained."""

        test_data = self.data.copy()

        test_data.loc[4, "proline"] = 10000

        results = audit_data(test_data)

        self.assertIn(
            "W005",
            results["retained"]["sample_id"].values
        )

        self.assertEqual(len(results["quarantine"]), 0)


if __name__ == "__main__":
    unittest.main()




if __name__ == "__main__":
    unittest.main()


    # Load the CSV dataset
    data_path = "wine_quality_case_PUBLIC_DERIVED.csv"

    data = pd.read_csv(data_path)

    # Run the audit
    results = audit_data(data)

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export reports
    results["retained"].to_csv(
        output_dir / "retained_flagged.csv",
        index=False
    )

    results["quarantine"].to_csv(
        output_dir / "quarantined_records.csv",
        index=False
    )

    results["summary"].to_csv(
        output_dir / "quality_summary.csv",
        index=False
    )

    print("Audit completed successfully!")
