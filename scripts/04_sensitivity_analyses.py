from __future__ import annotations

import pandas as pd
import numpy as np
import statsmodels.api as sm

from settings import OUTPUT_DERIVED, OUTPUT_TABLES


COVARIATES = ["high_ADI", "elix_count", "age_cat", "gender_collapsed", "culture_site_group", "current_care_setting"]


def prepare_design(df: pd.DataFrame, outcome: str) -> tuple[pd.Series, pd.DataFrame]:
    model_df = df[[outcome, *COVARIATES]].copy()
    model_df["current_care_setting"] = model_df["current_care_setting"].fillna("unknown")
    model_df["age_cat"] = model_df["age_cat"].fillna("Unknown")
    model_df["gender_collapsed"] = model_df["gender_collapsed"].fillna("Other_or_Unknown")
    model_df["culture_site_group"] = model_df["culture_site_group"].fillna("other")
    y = model_df[outcome].astype(float)
    X = pd.get_dummies(model_df.drop(columns=[outcome]), drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant="add").astype(float)
    return y, X


def fit_modified_poisson(df: pd.DataFrame, outcome: str, cluster: str | None = None) -> dict[str, float | str]:
    y, X = prepare_design(df, outcome)
    model = sm.GLM(y, X, family=sm.families.Poisson())
    if cluster is None:
        result = model.fit(cov_type="HC0")
    else:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    coef = result.params["high_ADI"]
    se = result.bse["high_ADI"]
    return {
        "outcome": outcome,
        "sample": "all_episodes_clustered" if cluster else "first_episode",
        "adjusted_RR": float(np.exp(coef)),
        "RR_95CI_low": float(np.exp(coef - 1.96 * se)),
        "RR_95CI_high": float(np.exp(coef + 1.96 * se)),
    }


if __name__ == "__main__":
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    first_df = pd.read_csv(OUTPUT_DERIVED / "analysis_dataset_first_episode.csv", low_memory=False)
    all_df = pd.read_csv(OUTPUT_DERIVED / "analysis_dataset_all_episodes.csv", low_memory=False)

    results = [
        fit_modified_poisson(first_df, "outcome_nonsusceptible"),
        fit_modified_poisson(first_df, "outcome_resistant_only"),
        fit_modified_poisson(first_df, "outcome_cro_nonsusceptible"),
        fit_modified_poisson(all_df, "outcome_nonsusceptible", cluster="anon_id"),
        fit_modified_poisson(all_df, "outcome_resistant_only", cluster="anon_id"),
        fit_modified_poisson(all_df, "outcome_cro_nonsusceptible", cluster="anon_id"),
    ]

    pd.DataFrame(results).to_csv(OUTPUT_TABLES / "sensitivity_total_effects.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))
