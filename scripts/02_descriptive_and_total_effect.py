from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from settings import OUTPUT_DERIVED, OUTPUT_TABLES


pd.options.mode.copy_on_write = True
BOOTSTRAP_N = 300
RNG_SEED = 20260418


def smd_binary(x: pd.Series, g: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    g = pd.to_numeric(g, errors="coerce").fillna(0)
    g1 = x.loc[g == 1].mean()
    g0 = x.loc[g == 0].mean()
    p = (g1 + g0) / 2
    denom = np.sqrt(max(p * (1 - p), 1e-12))
    return float((g1 - g0) / denom)


def smd_multilevel(series: pd.Series, group: pd.Series) -> float:
    dummies = pd.get_dummies(series.fillna("Missing"), prefix="cat", dtype=int)
    return float(max(abs(smd_binary(dummies[col], group)) for col in dummies.columns)) if len(dummies.columns) else np.nan


def median_iqr(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NA"
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}, {s.quantile(0.75):.1f}]"


def format_counts(series: pd.Series) -> dict[str, str]:
    out = {}
    denom = len(series)
    for key, value in series.fillna("Missing").value_counts(dropna=False).items():
        out[str(key)] = f"{int(value)} ({100 * value / denom:.1f}%)"
    return out


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group = df["high_ADI"]
    high = df.loc[group == 1]
    low = df.loc[group == 0]

    rows.append(
        {
            "variable": "elix_count",
            "high_ADI": median_iqr(high["elix_count"]),
            "low_ADI": median_iqr(low["elix_count"]),
            "SMD": round(abs(smd_binary((df["elix_count"] > df["elix_count"].median()).astype(int), group)), 3),
        }
    )

    categorical_vars = [
        "age_cat",
        "gender_collapsed",
        "culture_site_group",
        "current_care_setting",
        "organism",
        "any_abx_90",
        "nh_90",
        "any_invasive_proc_30",
        "class_n_90_cat",
    ]
    for var in categorical_vars:
        rows.append(
            {
                "variable": var,
                "high_ADI": json.dumps(format_counts(high[var]), ensure_ascii=False),
                "low_ADI": json.dumps(format_counts(low[var]), ensure_ascii=False),
                "SMD": round(abs(smd_multilevel(df[var], group)), 3),
            }
        )
    return pd.DataFrame(rows)


def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    defs = [
        ("high_ADI", [0, 1]),
        ("any_abx_90", [0, 1]),
        ("nh_90", [0, 1]),
        ("any_invasive_proc_30", [0, 1]),
    ]
    rows = []
    for var, levels in defs:
        for level in levels:
            sub = df.loc[df[var] == level]
            rows.append(
                {
                    "variable": var,
                    "level": level,
                    "n": int(len(sub)),
                    "outcome_rate": float(sub["outcome_nonsusceptible"].mean()) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def prepare_model_df(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    model_df = df[
        [
            "outcome_nonsusceptible",
            "high_ADI",
            "elix_count",
            "age_cat",
            "gender_collapsed",
            "culture_site_group",
            "current_care_setting",
        ]
    ].copy()
    model_df["current_care_setting"] = model_df["current_care_setting"].fillna("unknown")
    model_df["age_cat"] = model_df["age_cat"].fillna("Unknown")
    model_df["gender_collapsed"] = model_df["gender_collapsed"].fillna("Other_or_Unknown")
    model_df["culture_site_group"] = model_df["culture_site_group"].fillna("other")
    setting_order = ["outpatient", "inpatient", "emergency room", "urgent care", "day surgery", "unknown"]
    model_df["current_care_setting"] = pd.Categorical(
        model_df["current_care_setting"],
        categories=setting_order,
        ordered=False,
    )
    X = pd.get_dummies(model_df.drop(columns=["outcome_nonsusceptible"]), drop_first=True, dtype=float)
    keep_cols = [col for col in X.columns if X[col].nunique(dropna=False) > 1]
    X = X[keep_cols]
    X = sm.add_constant(X, has_constant="add").astype(float)
    y = model_df["outcome_nonsusceptible"].astype(float)
    return y, X


def fit_poisson_from_xy(y: pd.Series, X: pd.DataFrame):
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
    return poisson_model.fit(cov_type="HC0")


def compute_standardized_rd(result, X: pd.DataFrame) -> float:
    X1 = X.copy()
    X0 = X.copy()
    X1["high_ADI"] = 1.0
    X0["high_ADI"] = 0.0
    risk1 = result.predict(X1)
    risk0 = result.predict(X0)
    return float(np.mean(risk1 - risk0))


def bootstrap_rd(df: pd.DataFrame, n_boot: int = BOOTSTRAP_N, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rds = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx].copy()
        y_b, X_b = prepare_model_df(sample)
        try:
            result_b = fit_poisson_from_xy(y_b, X_b)
            rd_b = compute_standardized_rd(result_b, X_b)
            if np.isfinite(rd_b):
                rds.append(rd_b)
        except Exception:
            continue
    if len(rds) < max(50, n_boot // 3):
        raise RuntimeError(f"Too few successful bootstrap replicates: {len(rds)}")
    return float(np.quantile(rds, 0.025)), float(np.quantile(rds, 0.975))


def fit_modified_poisson(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y, X = prepare_model_df(df)
    result = fit_poisson_from_xy(y, X)

    coef = result.params["high_ADI"]
    se = result.bse["high_ADI"]
    rr = float(np.exp(coef))
    rr_lcl = float(np.exp(coef - 1.96 * se))
    rr_ucl = float(np.exp(coef + 1.96 * se))
    rd = compute_standardized_rd(result, X)
    rd_lcl, rd_ucl = bootstrap_rd(df)

    summary = pd.DataFrame(
        [
            {
                "contrast": "high_ADI vs low_ADI",
                "adjusted_RR": rr,
                "RR_95CI_low": rr_lcl,
                "RR_95CI_high": rr_ucl,
                "adjusted_RD": rd,
                "RD_95CI_low": rd_lcl,
                "RD_95CI_high": rd_ucl,
                "bootstrap_reps": BOOTSTRAP_N,
            }
        ]
    )
    coeffs = result.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    return summary, coeffs


if __name__ == "__main__":
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUTPUT_DERIVED / "analysis_dataset_first_episode.csv", low_memory=False)
    table1 = build_table1(df)
    table2 = build_table2(df)
    total_effect, coeffs = fit_modified_poisson(df)

    table1.to_csv(OUTPUT_TABLES / "table1_baseline.csv", index=False)
    table2.to_csv(OUTPUT_TABLES / "table2_crude_rates.csv", index=False)
    total_effect.to_csv(OUTPUT_TABLES / "total_effect_results.csv", index=False)
    coeffs.to_csv(OUTPUT_TABLES / "modified_poisson_coefficients.csv", index=False)
    print(total_effect.to_string(index=False))
