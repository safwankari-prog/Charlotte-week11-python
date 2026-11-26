"""Week 11 Class 3 student lab: implement a leakage-safe feature pipeline.
Run: python preprocess_iris.py --input iris_preprocessing_case_PUBLIC_DERIVED.csv --output-dir outputs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

CATEGORICAL = [
    "collection_site",
    "instrument_type",
]

LABEL = "species"


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC),
            ("cat", categorical_transformer, CATEGORICAL),
        ]
    )


def prepare_dataset(input_csv):
    df = pd.read_csv(input_csv)

    missing_source_cells = int(df.isna().sum().sum())

    X = df[NUMERIC + CATEGORICAL]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    transformer = build_preprocessor()

    train_array = transformer.fit_transform(X_train)
    test_array = transformer.transform(X_test)

    if not np.isfinite(train_array).all():
        raise ValueError("Non-finite values found in training output")

    if not np.isfinite(test_array).all():
        raise ValueError("Non-finite values found in test output")

    feature_names = transformer.get_feature_names_out()

    train_df = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=X_train.index,
    )

    test_df = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=X_test.index,
    )

    report = {
        "source_rows": len(df),
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "transformed_features": len(feature_names),
        "missing_source_cells": missing_source_cells,
    }

    return transformer, train_df, test_df, report


def write_outputs(train_df, test_df, report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train_features.csv", index=False)
    test_df.to_csv(output_dir / "test_features.csv", index=False)

    pd.DataFrame([report]).to_csv(
        output_dir / "data_quality_summary.csv",
        index=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="iris_preprocessing_case_PUBLIC_DERIVED.csv",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
    )

    args = p.parse_args()

    transformer, train, test, report = prepare_dataset(args.input)

    write_outputs(
        train,
        test,
        report,
        args.output_dir,
    )

    print("Preprocessing completed")


if __name__ == "__main__":
    main()
