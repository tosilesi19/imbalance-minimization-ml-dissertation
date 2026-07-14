# Evaluating the Interaction Between Class Imbalance and Data Minimization in Machine Learning Risk Outcomes

Computational experiment and analysis code for a PhD dissertation in
Information Technology, University of the Cumberlands.

**Author:** Tayo Osilesi | [Website](https://YOURSITE.com) | [LinkedIn](www.linkedin.com/in/tayo-osilesi-mba-cissp-a2036526)

## Overview

Regulatory frameworks such as the GDPR and CCPA mandate data minimization,
while real-world classification problems in finance and healthcare exhibit
severe class imbalance. This study examines what happens when these two
forces meet: a 4x4 full factorial computational experiment crossing four
class imbalance ratios (50/50, 80/20, 90/10, 95/5) with four data
minimization levels (100%, 75%, 50%, 25% retention), executed across two
public benchmark datasets, three classifier families, two minimization
strategies, and five random seeds: 960 verified run-seed combinations.

## Research Questions

1. **RQ1:** Does class imbalance ratio interact with data minimization
   level in determining classification performance?
2. **RQ2:** Does the interaction extend to fairness outcomes (equal
   opportunity difference, false negative rate difference) for the
   protected gender subgroup?
3. **RQ3:** Do standard and imbalance-sensitive evaluation metrics differ
   in their sensitivity to these effects?

## Design Summary

| Factor | Levels |
|---|---|
| Class imbalance ratio | 50/50, 80/20, 90/10, 95/5 |
| Data minimization level | 100%, 75%, 50%, 25% |
| Minimization type | Horizontal (row-wise), Vertical (feature-wise, MI-ranked) |
| Classifier | Logistic Regression, Random Forest, XGBoost |
| Dataset | UCI Adult Income (n=32,561, 42 features), UCI Diabetes 130-US Hospitals (n=101,766, 108 features) |
| Seeds | 11, 22, 33, 44, 55 |

Dependent variables: accuracy, minority-class recall, macro F1, balanced
accuracy, Matthews Correlation Coefficient, equal opportunity difference,
and false negative rate difference (female minus male convention).
Analysis: two-way factorial ANOVA per metric with Tukey HSD post hoc and
Welch robustness checks.

## Repository Structure

    src/          Importable modules: manipulations, training, metrics
    notebooks/    Numbered end-to-end pipeline (00 environment to 09 audit)
    results/      results_master.csv (960 runs), design grid, analysis outputs
    figures/      Publication figures (300 DPI, grayscale-safe)
    data/         Data source documentation (raw data not redistributed)
    docs/         Data dictionary and supporting documentation

## Reproducibility

1. **Environment:** `pip install -r requirements.txt` (versions frozen at
   execution time)
2. **Data:** download the two UCI datasets (links in `data/README.md`),
   run notebooks 01 and 02; assertion gates enforce the published dataset
   characteristics
3. **Execution:** notebook 06 re-runs any or all of the 960 conditions;
   all manipulations and models are seed-deterministic
4. **Audit:** notebook 09 re-executes a fixed random sample of 10 runs
   from archived code and data; all 70 metric values reproduce the master
   table exactly

## Data Sources

- Kohavi, R. (1996). Adult (Census Income). UCI Machine Learning Repository.
- Strack, B., et al. (2014). Diabetes 130-US Hospitals for years
  1999-2008. UCI Machine Learning Repository.

Both are public benchmark resources; this repository does not
redistribute them.

## Key Findings

*To be summarized following dissertation defense.*

## License

Code is released under the MIT License (see `LICENSE`). The dissertation
text is under separate copyright.