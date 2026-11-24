"""Week 11 Class 2 independent lab — complete the TODOs yourself."""
from pathlib import Path
import pandas as pd

REQUIRED = ["sample_id", "alcohol", "malic_acid", "color_intensity", "proline", "target"]

def audit_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return retained, quarantine and quality summary tables."""

    # Validate schema
    missing_columns = [
        col for col in REQUIRED
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Preserve original data
    df = data.copy()

    # Calculate statistical flags
    for column in ["malic_acid", "proline"]:

        values = pd.to_numeric(
            df[column], errors="coerce"
        )

        # Z-score
        mean = values.mean()
        std = values.std(ddof=0)

        if pd.notna(std) and std > 0:
            z_score = (values - mean) / std
            df[f"{column}_zscore_flag"] = (
                z_score.abs() > 3
            )
        else:
            df[f"{column}_zscore_flag"] = False

        # IQR
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[f"{column}_iqr_flag"] = (
            (values < lower) | (values > upper)
        ).fillna(False)

    # Domain validation
    malic = pd.to_numeric(df["malic_acid"], errors="coerce")
    proline = pd.to_numeric(df["proline"], errors="coerce")

    df["malic_acid_invalid_flag"] = (
        malic.isna() | (malic < 0)
    )

    df["proline_invalid_flag"] = (
        (proline <= 0).fillna(False)
    )

    # Combine statistical and domain flags into a review decision.
    flag_columns = [
        "malic_acid_zscore_flag",
        "malic_acid_iqr_flag",
        "proline_zscore_flag",
        "proline_iqr_flag",
        "malic_acid_invalid_flag",
        "proline_invalid_flag",
    ]
    df["review_needed"] = df[flag_columns].any(axis=1)

    retained = df.loc[~df["review_needed"]].copy()
    quarantine = df.loc[df["review_needed"]].copy()
    summary = pd.DataFrame(
        {
            "metric": ["total_records", "retained_records", "quarantined_records"],
            "count": [len(df), len(retained), len(quarantine)],
        }
    )

    return {
        "retained": retained,
        "quarantine": quarantine,
        "summary": summary,
    }

if __name__ == "__main__":
    data = pd.read_csv(Path(__file__).with_name("wine_quality_case_PUBLIC_DERIVED.csv"))
    results = audit_data(data)
    # TODO: save all three CSVs with documented relative path

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
