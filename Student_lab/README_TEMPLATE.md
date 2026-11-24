# Week 11 Class 2 — Outlier Audit

Source: UCI Wine dataset distributed with scikit-learn. Modified for education with three instructor-inserted anomalies.

## Run

1. Install required Python dependencies.
2. Execute the audit workflow:

```bash
python outlier_audit.py
```

3. Review the generated outputs:

- retained_flagged.csv
- quarantined_records.csv
- quality_summary.csv

## Validation and Decision Policy

The audit combines statistical outlier detection and data-quality validation checks.

Methods used:

- **Z-score detection** to identify extreme values far from the population mean.
- **Interquartile Range (IQR) detection** to identify values outside expected distribution boundaries.
- **Validation checks** to identify impossible or invalid values.

Records that trigger one or more rules are flagged for review rather than automatically removed. Statistical outliers can represent legitimate observations, while validation failures may indicate data-entry errors or corrupted values. Retaining flagged observations until review helps preserve transparency and prevents accidental loss of valid data.

## Results and Limitations

### Results

The audit processed **178 total wine records**.

Quality summary:

| Category | Count |
|----------|------:|
| Total Records | 178 |
| Retained Records | 172 |
| Quarantined Records | 6 |

The retained dataset contains observations that passed validation and did not require manual review. The quarantine dataset contains records flagged by one or more validation or outlier-detection rules.

### Quarantined Records

| Sample ID | Reason for Review |
|------------|------------------|
| WINE-013 | Invalid negative proline value (-45.0) |
| WINE-062 | Malic acid value (99.0) flagged by both Z-score and IQR methods |
| WINE-124 | Malic acid value flagged by IQR method |
| WINE-131 | Proline value (9999.0) flagged by both Z-score and IQR methods |
| WINE-138 | Malic acid value flagged by IQR method |
| WINE-174 | Malic acid value flagged by IQR method |

These results demonstrate that the quality-control pipeline successfully detected both obvious validation failures and statistical anomalies requiring human review.

### Limitations

- Outlier detection methods can generate false positives.
- IQR and Z-score thresholds are sensitive to the underlying data distribution.
- Statistical methods alone cannot determine whether an outlier is an error or a valid rare observation.
- Manual review remains necessary before final removal decisions are made.

## GitHub PR

This pull request includes:

- Data validation checks for impossible values.
- Outlier detection using Z-score and IQR approaches.
- Generation of retained and quarantined datasets.
- Quantitative quality summary reporting.
- Documentation describing review and retention policies.

The project follows a review-first approach: anomalous records are quarantined and documented rather than automatically deleted, ensuring reproducibility and auditability of data-cleaning decisions.
