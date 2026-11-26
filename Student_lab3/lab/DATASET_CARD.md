# Dataset card — Week 11 Class 3
**Public measured portion:** Iris flowers; four numerical measurements and species labels, bundled by scikit-learn (`load_iris`, originally the UCI Iris dataset). Source: https://archive.ics.uci.edu/dataset/53/iris and https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
**Changes made for teaching:** Added `sample_id` (synthetic), `collection_site` (synthetic), `instrument_type` (synthetic), inserted seven missing numeric observations and four missing site values. The added categories were NOT collected in the original research and should not be interpreted as measured botanical facts. The original 150 numeric rows are retained otherwise.
**Purpose:** Predictors preparation only. `species` is the label, never a predictor. This dataset is not a deployment-ready scientific model.
**Formats:** Student version distributed as UTF-8 CSV; `.xlsx`/JSON/Parquet examples are in Student Notes, not additional compulsory assignments.
**Privacy/licensing:** No personal records; cite the Iris dataset and scikit-learn provenance when using this training case.
