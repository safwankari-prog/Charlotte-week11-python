# Dataset card — Week 11 Class 4

**Main lab sources:** `well_b_synthetic.las` and `scada_synthetic.csv` are fully SYNTHETIC educational data. The LAS file follows a simplified LAS 2.0 structure. `depth_m` is a deliberately shared key to simplify the introduction of cross-source merges; operational SCADA usually uses timestamps and production well logs often need an explicit depth/time reconciliation policy.

**Other formats:** `sample_api_payload.json` is synthetic; `iris_public_subset.csv` is a subset of the public Fisher Iris measurements from the UCI Iris dataset, made available through scikit-learn. `teaching/iris_public_uci_derived.csv` preserves original numeric measurements and adds species names from the dataset metadata; it is not industrial field data.

Original public dataset: https://archive.ics.uci.edu/dataset/53/iris . Check licensing/source notes before redistributing outside the training setting. No individual personal data appears in these files.

**Data issues:** `scada_synthetic.csv` contains missing pressure and flow entries. The LAS data is illustrative and not certified for engineering decisions. Never treat synthetic values as real production measurements.

**Learner expectation:** validate merge coverage, preserve missingness indicators, calculate new features without using future observations, and explicitly document any imputation policy. Never fill a pressure or flow missing value with zero without a reason.
