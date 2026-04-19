from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from settings import MC_MED_ROOT, OUTPUT_DERIVED, OUTPUT_TABLES


@dataclass
class MediationResult:
    block: str
    total_effect_rd: float
    indirect_effect_rd: float
    direct_effect_rd: float
    proportion_mediated: float


MEDIATION_BLOCKS = {
    "abx_pressure": ["any_abx_90", "class_n_90_cat"],
    "healthcare_exposure": ["nh_90", "any_invasive_proc_30"],
    "combined": ["any_abx_90", "class_n_90_cat", "nh_90", "any_invasive_proc_30"],
}

COVARIATES = ["age_cat", "gender_collapsed", "culture_site_group", "current_care_setting", "elix_count"]


def prepare_matrix(df: pd.DataFrame, mediators: Iterable[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = ["high_ADI", "outcome_nonsusceptible", *mediators, *COVARIATES]
    model_df = df[cols].copy()
    X = pd.get_dummies(model_df.drop(columns=["outcome_nonsusceptible"]), drop_first=True)
    y = model_df["outcome_nonsusceptible"].astype(int)
    a = model_df["high_ADI"].astype(int)
    return X, y, a


def estimate_total_effect(df: pd.DataFrame) -> float:
    return float(df.loc[df["high_ADI"] == 1, "outcome_nonsusceptible"].mean() - df.loc[df["high_ADI"] == 0, "outcome_nonsusceptible"].mean())


def estimate_interventional_disparity(df: pd.DataFrame, mediators: list[str], random_state: int = 42) -> MediationResult:
    work = df.copy()
    total_effect = estimate_total_effect(work)

    mediator_means_low = {}
    for mediator in mediators:
        if work[mediator].dtype == object:
            mediator_means_low[mediator] = work.loc[work["high_ADI"] == 0, mediator].mode(dropna=True).iat[0]
        else:
            mediator_means_low[mediator] = work.loc[work["high_ADI"] == 0, mediator].mean()

    intervened = work.copy()
    for mediator, value in mediator_means_low.items():
        intervened.loc[intervened["high_ADI"] == 1, mediator] = value

    X_obs, y_obs, _ = prepare_matrix(work, mediators)
    X_int, _, _ = prepare_matrix(intervened, mediators)

    model = HistGradientBoostingClassifier(random_state=random_state)
    model.fit(X_obs, y_obs)
    risk_obs = model.predict_proba(X_obs)[:, 1]
    risk_int = model.predict_proba(X_int)[:, 1]

    disparity_after_shift = float(risk_int[work["high_ADI"] == 1].mean() - risk_obs[work["high_ADI"] == 0].mean())
    indirect_effect = total_effect - disparity_after_shift
    proportion = indirect_effect / total_effect if total_effect != 0 else np.nan
    return MediationResult(
        block=" + ".join(mediators),
        total_effect_rd=total_effect,
        indirect_effect_rd=indirect_effect,
        direct_effect_rd=disparity_after_shift,
        proportion_mediated=proportion,
    )


def build_sensitivity_manifest() -> pd.DataFrame:
    files = sorted(path.name for path in MC_MED_ROOT.glob("*.csv"))
    return pd.DataFrame({"mc_med_file": files})


if __name__ == "__main__":
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUTPUT_DERIVED / "analysis_dataset_first_episode.csv", low_memory=False)
    results = []
    for name, mediators in MEDIATION_BLOCKS.items():
        result = estimate_interventional_disparity(df, mediators)
        results.append(result.__dict__ | {"block_name": name})
    pd.DataFrame(results).to_csv(OUTPUT_TABLES / "mediation_results_scaffold.csv", index=False)
    build_sensitivity_manifest().to_csv(OUTPUT_TABLES / "mc_med_manifest.csv", index=False)
    print(json.dumps(results, indent=2))
