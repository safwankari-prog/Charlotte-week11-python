"""Week 11 Class 4 — Feature engineering from LAS and SCADA data."""
from pathlib import Path

import lasio
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_scada(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = [
        "well_id",
        "depth_m",
        "pressure_psi",
        "temperature_c",
        "flow_m3_h",
        "operating_mode",
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing SCADA columns: {missing_columns}")

    numeric_columns = [
        "depth_m",
        "pressure_psi",
        "temperature_c",
        "flow_m3_h",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["well_id", "depth_m"]].isna().any().any():
        raise ValueError("SCADA well_id and depth_m cannot be missing.")

    if df.duplicated(["well_id", "depth_m"]).any():
        raise ValueError("Duplicate SCADA well/depth keys found.")

    return df.sort_values(["well_id", "depth_m"]).reset_index(drop=True)
def load_las(path: Path) -> pd.DataFrame:
    las = lasio.read(str(path))

    well_id = str(las.well.WELL.value).strip()

    null_value = las.well.NULL.value
    df = las.df().reset_index()

    required_curves = ["DEPT", "GR", "RHOB"]
    missing_curves = set(required_curves) - set(df.columns)

    if missing_curves:
        raise ValueError(f"Missing LAS curves: {missing_curves}")

    df = df.rename(
        columns={
            "DEPT": "depth_m",
            "GR": "gamma_api",
            "RHOB": "density_g_cm3",
        }
    )

    df["well_id"] = well_id

    numeric_columns = ["depth_m", "gamma_api", "density_g_cm3"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[numeric_columns] = df[numeric_columns].replace(null_value, np.nan)

    if df[["well_id", "depth_m"]].isna().any().any():
        raise ValueError("LAS well_id and depth_m cannot be missing.")

    if df.duplicated(["well_id", "depth_m"]).any():
        raise ValueError("Duplicate LAS well/depth keys found.")

    return df[["well_id", "depth_m", "gamma_api", "density_g_cm3"]]

def merge_inputs(las_df: pd.DataFrame, scada_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["well_id", "depth_m"]

    if las_df.duplicated(keys).any():
        raise ValueError("Duplicate LAS merge keys found.")

    if scada_df.duplicated(keys).any():
        raise ValueError("Duplicate SCADA merge keys found.")

    merged = pd.merge(
        scada_df,
        las_df,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]

    if not unmatched.empty:
        raise ValueError(
            f"Unmatched well/depth rows found:\n{unmatched[keys + ['_merge']]}"
        )

    return (
        merged.drop(columns="_merge")
        .sort_values(keys)
        .reset_index(drop=True)
    )
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["well_id", "depth_m"]).reset_index(drop=True)

    grouped = df.groupby("well_id", sort=False)

    df["pressure_missing_flag"] = df["pressure_psi"].isna().astype(int)
    df["flow_missing_flag"] = df["flow_m3_h"].isna().astype(int)

    df["pressure_delta_psi"] = grouped["pressure_psi"].diff()

    df["pressure_var_3"] = grouped["pressure_psi"].transform(
        lambda values: values.rolling(window=3, min_periods=3).var(ddof=0)
    )
    df["temperature_delta_c"] = grouped["temperature_c"].diff()

    df["cumulative_sampled_flow"] = grouped["flow_m3_h"].cumsum()

    df["gamma_density_ratio"] = (
        df["gamma_api"] / df["density_g_cm3"]
    )

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    scada_df = load_scada(DATA_DIR / "scada_synthetic.csv")
    las_df = load_las(DATA_DIR / "well_b_synthetic.las")

    merged_df = merge_inputs(las_df, scada_df)
    features_df = engineer_features(merged_df)

    features_df.to_csv(
        OUTPUT_DIR / "engineered_features.csv",
        index=False,
    )

    quality_summary = pd.DataFrame(
        {
            "metric": [
                "scada_input_rows",
                "las_input_rows",
                "merged_output_rows",
                "pressure_missing_count",
                "flow_missing_count",
                "unmatched_merge_rows",
            ],
            "value": [
                len(scada_df),
                len(las_df),
                len(features_df),
                int(features_df["pressure_psi"].isna().sum()),
                int(features_df["flow_m3_h"].isna().sum()),
                0,
            ],
        }
    )

    quality_summary.to_csv(
        OUTPUT_DIR / "quality_summary.csv",
        index=False,
    )

    print("Feature pipeline completed.")
    print(quality_summary.to_string(index=False))


if __name__ == "__main__":
    main()
