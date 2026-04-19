# ARMD-MGB ESBL Mediation Analysis

## Purpose
This workspace builds the manuscript-ready cohort for the study:
Healthcare exposure as a mediator of socioeconomic disparities in extended-spectrum beta-lactam non-susceptibility among Enterobacterales.

## Data sources
Primary data are read from:
- C:\Users\adamk\Desktop\mimic-iv-3.1\antibiotic-resistance-microbiology-dataset-mass-general-brigham-armd-mgb-1.0.0\antibiotic-resistance-microbiology-dataset-mass-general-brigham-armd-mgb-1.0.0

Sensitivity manifests are read from:
- C:\Users\adamk\Desktop\mimic-iv-3.1\MC-MED

## Analysis structure
1. `scripts\01_build_analysis_dataset.py`
   Builds one row per culture-organism episode, restricts to first eligible episode per patient, and derives:
   - primary outcome
   - CRO-only outcome
   - resistant-only outcome
   - ADI exposure groups
   - antibiotic mediators
   - healthcare contact mediators
   - baseline covariates

2. `scripts\02_descriptive_and_total_effect.py`
   Produces:
   - Table 1 baseline characteristics with SMDs
   - Table 2 crude outcome rates
   - modified Poisson adjusted RR
   - standardized adjusted RD approximation

3. `scripts\03_mediation_scaffold.py`
   Produces a first-pass interventional disparity scaffold for:
   - antibiotic pressure block
   - healthcare exposure block
   - combined mediators

## Outputs
Derived datasets go to `output\derived`.
Tables go to `output\tables`.

## Notes
- Main microbiology file is `microbiology_cohort_deid_tj_updated.csv` in this local copy.
- ADI main analysis is restricted to state deciles 1 to 3 versus 8 to 10.
- Polymicrobial AST episodes are excluded in the main cohort.
- The mediation script is intentionally a scaffold and should be upgraded to bootstrap inference before manuscript submission.
