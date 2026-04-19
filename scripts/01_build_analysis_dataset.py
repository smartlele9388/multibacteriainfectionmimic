from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from settings import (
    ABX_CLASS_GROUPS,
    ARMD_ROOT,
    BROAD_SPECTRUM_CODES,
    CARE_SETTING_MAP,
    COMMON_USECOLS,
    INVASIVE_PROCEDURES,
    OUTPUT_DERIVED,
    PRIMARY_PHENO_POSITIVE,
    RESISTANT_ONLY,
    TARGET_ORGANISMS,
)


pd.options.mode.copy_on_write = True
CHUNKSIZE = 250_000


def ensure_output_dirs() -> None:
    OUTPUT_DERIVED.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_organism_name(value: object) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).upper())


def normalize_drug_class(value: object) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_episode_records() -> tuple[pd.DataFrame, dict[str, int]]:
    path = ARMD_ROOT / "microbiology_cohort_deid_tj_updated.csv"
    usecols = COMMON_USECOLS + [
        "order_time_jittered_utc_shifted",
        "culture_description",
        "organism",
        "neg_cx",
        "mult_org_ast",
        "has_AST",
        "prelim_AST",
        "AST_code",
        "CLSI_2022_pheno",
    ]
    records: dict[tuple[int, int, int, str], dict[str, object]] = {}
    flow = defaultdict(int)

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
        flow["raw_rows"] += len(chunk)
        chunk["organism_norm"] = chunk["organism"].map(normalize_organism_name)
        mask = (
            chunk["neg_cx"].fillna("").ne("X")
            & chunk["has_AST"].fillna("").eq("X")
            & chunk["prelim_AST"].fillna("").ne("X")
            & chunk["organism_norm"].isin(TARGET_ORGANISMS)
        )
        chunk = chunk.loc[mask].copy()
        flow["after_positive_has_ast_nonprelim_target_org_rows"] += len(chunk)
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            key = (row.anon_id, row.pat_enc_csn_id_coded, row.order_proc_id_coded, row.organism_norm)
            rec = records.get(key)
            if rec is None:
                rec = {
                    "anon_id": row.anon_id,
                    "pat_enc_csn_id_coded": row.pat_enc_csn_id_coded,
                    "order_proc_id_coded": row.order_proc_id_coded,
                    "order_time_jittered_utc_shifted": normalize_text(row.order_time_jittered_utc_shifted),
                    "culture_description": normalize_text(row.culture_description).lower(),
                    "organism": TARGET_ORGANISMS[row.organism_norm],
                    "mult_org_ast_flag": 0,
                    "drug_codes_present": set(),
                    "outcome_nonsusceptible": 0,
                    "outcome_resistant_only": 0,
                    "outcome_cro_nonsusceptible": 0,
                    "all_broad_susceptible": 1,
                    "has_any_broad_result": 0,
                }
                records[key] = rec
            if normalize_text(row.mult_org_ast) == "X":
                rec["mult_org_ast_flag"] = 1
            ast_code = normalize_text(row.AST_code)
            pheno = normalize_text(row.CLSI_2022_pheno)
            if ast_code in BROAD_SPECTRUM_CODES:
                rec["has_any_broad_result"] = 1
                rec["drug_codes_present"].add(ast_code)
                if pheno in PRIMARY_PHENO_POSITIVE:
                    rec["outcome_nonsusceptible"] = 1
                if pheno in RESISTANT_ONLY:
                    rec["outcome_resistant_only"] = 1
                if ast_code == "CRO" and pheno in PRIMARY_PHENO_POSITIVE:
                    rec["outcome_cro_nonsusceptible"] = 1
                if pheno != "Susceptible":
                    rec["all_broad_susceptible"] = 0

    rows = []
    for rec in records.values():
        rec = rec.copy()
        rec["drug_codes_present"] = "|".join(sorted(rec["drug_codes_present"]))
        rec["broad_drug_count"] = len(rec["drug_codes_present"].split("|")) if rec["drug_codes_present"] else 0
        rows.append(rec)

    episode = pd.DataFrame(rows)
    flow["culture_organism_episodes_before_polymicrobial_exclusion"] = len(episode)
    if episode.empty:
        return episode, dict(flow)

    episode = episode.loc[episode["mult_org_ast_flag"] == 0].copy()
    flow["after_polymicrobial_exclusion"] = len(episode)
    episode = episode.loc[episode["has_any_broad_result"] == 1].copy()
    flow["after_broad_spectrum_result_requirement"] = len(episode)
    return episode, dict(flow)


def load_adi() -> pd.DataFrame:
    adi = pd.read_csv(ARMD_ROOT / "ADI_deid_tj.csv", usecols=COMMON_USECOLS + ["adi_score", "adi_state_rank"], low_memory=False)
    adi = adi.drop_duplicates(subset=["order_proc_id_coded"])
    adi["adi_state_rank"] = pd.to_numeric(adi["adi_state_rank"], errors="coerce")
    adi["high_ADI"] = (adi["adi_state_rank"] >= 8) & (adi["adi_state_rank"] <= 10)
    adi["low_ADI"] = (adi["adi_state_rank"] >= 1) & (adi["adi_state_rank"] <= 3)
    adi["adi_analysis_group"] = np.select([adi["high_ADI"], adi["low_ADI"]], ["high", "low"], default="mid_or_missing")
    return adi


def load_demographics() -> pd.DataFrame:
    demo = pd.read_csv(ARMD_ROOT / "demographics_deid_tj.csv", usecols=COMMON_USECOLS + ["age", "gender"], low_memory=False)
    demo = demo.drop_duplicates(subset=["order_proc_id_coded"])
    demo["age_cat"] = demo["age"].map(normalize_text)
    demo["gender"] = demo["gender"].map(normalize_text).replace({"": "Unknown"})
    demo["gender_collapsed"] = demo["gender"].where(~demo["gender"].isin(["Other", "Unknown"]), "Other_or_Unknown")
    return demo[["order_proc_id_coded", "age_cat", "gender", "gender_collapsed"]]


def load_ward_type() -> pd.DataFrame:
    ward = pd.read_csv(ARMD_ROOT / "ward_type_deid_tj.csv", usecols=COMMON_USECOLS + list(CARE_SETTING_MAP), low_memory=False)
    ward = ward.drop_duplicates(subset=["order_proc_id_coded"])
    for col in CARE_SETTING_MAP:
        ward[col] = pd.to_numeric(ward[col], errors="coerce").fillna(0).astype(int)

    def to_setting(row: pd.Series) -> str:
        active = [CARE_SETTING_MAP[col] for col in CARE_SETTING_MAP if row[col] == 1]
        return active[0] if active else "unknown"

    ward["current_care_setting"] = ward.apply(to_setting, axis=1)
    return ward[["order_proc_id_coded", "current_care_setting"]]


def load_comorbidity() -> pd.DataFrame:
    path = ARMD_ROOT / "comorbidity_deid_tj.csv"
    counts = defaultdict(set)
    flags = defaultdict(lambda: {
        "elix_diabetes": 0,
        "elix_renal_failure": 0,
        "elix_chronic_pulmonary": 0,
        "elix_liver_disease": 0,
        "elix_cancer": 0,
    })

    for chunk in pd.read_csv(path, usecols=COMMON_USECOLS + ["category"], chunksize=CHUNKSIZE, low_memory=False):
        chunk["category_norm"] = chunk["category"].map(lambda x: normalize_text(x).lower())
        chunk = chunk.loc[chunk["category_norm"] != ""]
        for row in chunk.itertuples(index=False):
            opid = row.order_proc_id_coded
            cat = row.category_norm
            counts[opid].add(cat)
            if "diabetes" in cat:
                flags[opid]["elix_diabetes"] = 1
            if "renal failure" in cat or "kidney disease" in cat:
                flags[opid]["elix_renal_failure"] = 1
            if "chronic pulmonary" in cat or "copd" in cat:
                flags[opid]["elix_chronic_pulmonary"] = 1
            if "liver disease" in cat:
                flags[opid]["elix_liver_disease"] = 1
            if "cancer" in cat or "malignan" in cat or "tumor" in cat:
                flags[opid]["elix_cancer"] = 1

    rows = []
    all_opids = set(counts) | set(flags)
    for opid in all_opids:
        row = {"order_proc_id_coded": opid, "elix_count": len(counts.get(opid, set()))}
        row.update(flags.get(opid, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def load_abx_mediators() -> pd.DataFrame:
    path = ARMD_ROOT / "prior_abx_deid_tj.csv"
    summary = defaultdict(lambda: {
        "any_abx_30": 0,
        "any_abx_90": 0,
        "class_n_90_set": set(),
        "class_n_365_set": set(),
        **{key: 0 for key in ABX_CLASS_GROUPS},
    })

    for chunk in pd.read_csv(path, usecols=COMMON_USECOLS + ["last_dose_to_culture", "drug_class"], chunksize=CHUNKSIZE, low_memory=False):
        chunk["last_dose_to_culture"] = pd.to_numeric(chunk["last_dose_to_culture"], errors="coerce")
        chunk["drug_class_norm"] = chunk["drug_class"].map(normalize_drug_class)
        chunk = chunk.loc[chunk["last_dose_to_culture"].notna()]
        for row in chunk.itertuples(index=False):
            opid = row.order_proc_id_coded
            days = row.last_dose_to_culture
            drug_class = row.drug_class_norm
            rec = summary[opid]
            if days <= 30:
                rec["any_abx_30"] = 1
            if days <= 90:
                rec["any_abx_90"] = 1
                if drug_class:
                    rec["class_n_90_set"].add(drug_class)
                for col, group in ABX_CLASS_GROUPS.items():
                    if drug_class in group:
                        rec[col] = 1
            if days <= 365 and drug_class:
                rec["class_n_365_set"].add(drug_class)

    rows = []
    for opid, rec in summary.items():
        row = {
            "order_proc_id_coded": opid,
            "any_abx_30": rec["any_abx_30"],
            "any_abx_90": rec["any_abx_90"],
            "class_n_90": len(rec["class_n_90_set"]),
            "class_n_365": len(rec["class_n_365_set"]),
        }
        for col in ABX_CLASS_GROUPS:
            row[col] = rec[col]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["class_n_90_cat"] = np.select(
        [out["class_n_90"] == 0, out["class_n_90"] == 1, out["class_n_90"] >= 2],
        ["0", "1", ">=2"],
        default="0",
    )
    return out


def load_healthcare_mediators() -> pd.DataFrame:
    nh_path = ARMD_ROOT / "nursing_home_visits_deid_tj.csv"
    nh_summary = defaultdict(lambda: {"nh_30": 0, "nh_90": 0})
    for chunk in pd.read_csv(nh_path, usecols=COMMON_USECOLS + ["nursing_home_visit_culture"], chunksize=CHUNKSIZE, low_memory=False):
        chunk["nursing_home_visit_culture"] = pd.to_numeric(chunk["nursing_home_visit_culture"], errors="coerce")
        chunk = chunk.loc[chunk["nursing_home_visit_culture"].notna()]
        for row in chunk.itertuples(index=False):
            opid = row.order_proc_id_coded
            days = row.nursing_home_visit_culture
            if days <= 30:
                nh_summary[opid]["nh_30"] = 1
            if days <= 90:
                nh_summary[opid]["nh_90"] = 1

    proc_path = ARMD_ROOT / "prior_procedures_deid_tj.csv"
    proc_summary = defaultdict(lambda: {value: 0 for value in INVASIVE_PROCEDURES.values()} | {"any_invasive_proc_30": 0})
    for chunk in pd.read_csv(proc_path, usecols=COMMON_USECOLS + ["procedure_description", "procedure_days_culture"], chunksize=CHUNKSIZE, low_memory=False):
        chunk["procedure_days_culture"] = pd.to_numeric(chunk["procedure_days_culture"], errors="coerce")
        chunk["procedure_description"] = chunk["procedure_description"].map(lambda x: normalize_text(x).lower())
        chunk = chunk.loc[chunk["procedure_days_culture"].notna() & (chunk["procedure_days_culture"] <= 30)]
        for row in chunk.itertuples(index=False):
            opid = row.order_proc_id_coded
            if row.procedure_description in INVASIVE_PROCEDURES:
                col = INVASIVE_PROCEDURES[row.procedure_description]
                proc_summary[opid][col] = 1
                proc_summary[opid]["any_invasive_proc_30"] = 1

    all_opids = set(nh_summary) | set(proc_summary)
    rows = []
    for opid in all_opids:
        row = {"order_proc_id_coded": opid}
        row.update(nh_summary.get(opid, {"nh_30": 0, "nh_90": 0}))
        row.update(proc_summary.get(opid, {value: 0 for value in INVASIVE_PROCEDURES.values()} | {"any_invasive_proc_30": 0}))
        rows.append(row)
    return pd.DataFrame(rows)


def load_prior_micro_history() -> pd.DataFrame:
    path = ARMD_ROOT / "prior_micro_deid_tj.csv"
    summary = defaultdict(lambda: {"prior_resistant_micro_365": 0, "prior_nonsus_micro_365": 0})
    for chunk in pd.read_csv(path, usecols=COMMON_USECOLS + ["drug_code", "CLSI_2022_pheno", "prior_AST_time_to_culture"], chunksize=CHUNKSIZE, low_memory=False):
        chunk["prior_AST_time_to_culture"] = pd.to_numeric(chunk["prior_AST_time_to_culture"], errors="coerce")
        chunk = chunk.loc[chunk["prior_AST_time_to_culture"].notna() & (chunk["prior_AST_time_to_culture"] <= 365)]
        if chunk.empty:
            continue
        chunk["drug_code"] = chunk["drug_code"].map(normalize_text)
        chunk["CLSI_2022_pheno"] = chunk["CLSI_2022_pheno"].map(normalize_text)
        chunk = chunk.loc[chunk["drug_code"].isin(BROAD_SPECTRUM_CODES)]
        for row in chunk.itertuples(index=False):
            opid = row.order_proc_id_coded
            if row.CLSI_2022_pheno in PRIMARY_PHENO_POSITIVE:
                summary[opid]["prior_nonsus_micro_365"] = 1
            if row.CLSI_2022_pheno in RESISTANT_ONLY:
                summary[opid]["prior_resistant_micro_365"] = 1
    return pd.DataFrame([{"order_proc_id_coded": k, **v} for k, v in summary.items()])


def load_prior_org_history() -> pd.DataFrame:
    path = ARMD_ROOT / "prior_org_deid_tj.csv"
    summary = defaultdict(lambda: {"prior_target_org_365": 0})
    target_norm = set(TARGET_ORGANISMS.keys())
    for chunk in pd.read_csv(path, usecols=COMMON_USECOLS + ["prior_org_days_to_culture", "prior_org"], chunksize=CHUNKSIZE, low_memory=False):
        chunk["prior_org_days_to_culture"] = pd.to_numeric(chunk["prior_org_days_to_culture"], errors="coerce")
        chunk = chunk.loc[chunk["prior_org_days_to_culture"].notna() & (chunk["prior_org_days_to_culture"] <= 365)]
        if chunk.empty:
            continue
        chunk["prior_org_norm"] = chunk["prior_org"].map(normalize_organism_name)
        for row in chunk.itertuples(index=False):
            if row.prior_org_norm in target_norm:
                summary[row.order_proc_id_coded]["prior_target_org_365"] = 1
    return pd.DataFrame([{"order_proc_id_coded": k, **v} for k, v in summary.items()])


def assemble_analysis_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    episode, flow = build_episode_records()
    adi = load_adi()
    demo = load_demographics()
    ward = load_ward_type()
    comorb = load_comorbidity()
    abx = load_abx_mediators()
    healthcare = load_healthcare_mediators()
    prior_micro = load_prior_micro_history()
    prior_org = load_prior_org_history()

    analysis = (
        episode.merge(adi, on=["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"], how="left")
        .merge(demo, on="order_proc_id_coded", how="left")
        .merge(ward, on="order_proc_id_coded", how="left")
        .merge(comorb, on="order_proc_id_coded", how="left")
        .merge(abx, on="order_proc_id_coded", how="left")
        .merge(healthcare, on="order_proc_id_coded", how="left")
        .merge(prior_micro, on="order_proc_id_coded", how="left")
        .merge(prior_org, on="order_proc_id_coded", how="left")
    )
    flow["after_joining_covariates"] = len(analysis)

    analysis["order_time_jittered_utc_shifted"] = pd.to_datetime(analysis["order_time_jittered_utc_shifted"], errors="coerce")
    analysis = analysis.sort_values(["anon_id", "order_time_jittered_utc_shifted", "order_proc_id_coded"])

    fill_zero_cols = [
        "elix_count", "any_abx_30", "any_abx_90", "class_n_90", "class_n_365", "nh_30", "nh_90",
        "cvc_30", "dialysis_30", "foley_30", "mechvent_30", "surgery_30", "any_invasive_proc_30",
        "prior_resistant_micro_365", "prior_nonsus_micro_365", "prior_target_org_365",
        *ABX_CLASS_GROUPS.keys(),
    ]
    for col in fill_zero_cols:
        if col in analysis.columns:
            analysis[col] = analysis[col].fillna(0)
    if "class_n_90_cat" in analysis.columns:
        analysis["class_n_90_cat"] = analysis["class_n_90_cat"].fillna("0")

    for col in [c for c in analysis.columns if c.startswith("elix_") and c != "elix_count"]:
        analysis[col] = analysis[col].fillna(0).astype(int)

    analysis["high_ADI"] = analysis["adi_analysis_group"].eq("high").astype(int)
    analysis["culture_site_group"] = np.select(
        [
            analysis["culture_description"].str.contains("urine", na=False),
            analysis["culture_description"].str.contains("blood", na=False),
            analysis["culture_description"].str.contains("resp", na=False),
        ],
        ["urine", "blood", "respiratory"],
        default="other",
    )
    analysis["setting_binary"] = np.where(analysis["current_care_setting"].eq("inpatient"), "inpatient", "non_inpatient")
    analysis["organism_group"] = np.select(
        [
            analysis["organism"].eq("Escherichia coli"),
            analysis["organism"].str.contains("Klebsiella", na=False),
        ],
        ["E. coli", "Klebsiella"],
        default="Other Enterobacterales",
    )
    return analysis, flow


def build_analysis_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_adi_episodes, flow = assemble_analysis_dataset()
    flow["after_adi_nonmissing"] = int(all_adi_episodes["adi_state_rank"].notna().sum())
    restricted_all_episodes = all_adi_episodes.loc[all_adi_episodes["adi_analysis_group"].isin(["high", "low"])].copy()
    flow["after_adi_high_low_restriction"] = len(restricted_all_episodes)

    all_adi_first = all_adi_episodes.loc[all_adi_episodes["adi_state_rank"].notna()].copy()
    all_adi_first["episode_rank_patient_all_adi"] = all_adi_first.groupby("anon_id").cumcount() + 1
    all_adi_first = all_adi_first.loc[all_adi_first["episode_rank_patient_all_adi"] == 1].copy()

    restricted_all_episodes["episode_rank_patient"] = restricted_all_episodes.groupby("anon_id").cumcount() + 1
    first_episode = restricted_all_episodes.loc[restricted_all_episodes["episode_rank_patient"] == 1].copy()
    flow["first_episode_per_patient"] = len(first_episode)
    flow_df = pd.DataFrame({"step": list(flow.keys()), "n": list(flow.values())})
    return all_adi_episodes, all_adi_first, restricted_all_episodes, first_episode, flow_df


if __name__ == "__main__":
    ensure_output_dirs()
    all_adi_episodes, all_adi_first, all_episodes, first_episode, flow_df = build_analysis_dataset()
    all_adi_episodes.to_csv(OUTPUT_DERIVED / "analysis_dataset_all_adi_episodes.csv", index=False)
    all_adi_first.to_csv(OUTPUT_DERIVED / "analysis_dataset_all_adi_first_episode.csv", index=False)
    all_episodes.to_csv(OUTPUT_DERIVED / "analysis_dataset_all_episodes.csv", index=False)
    first_episode.to_csv(OUTPUT_DERIVED / "analysis_dataset_first_episode.csv", index=False)
    flow_df.to_csv(OUTPUT_DERIVED / "flow_counts.csv", index=False)
    summary = {
        "n_analysis_all_adi_episodes": int(len(all_adi_episodes)),
        "n_nonmissing_adi_episodes": int(all_adi_episodes["adi_state_rank"].notna().sum()),
        "n_analysis_all_episodes": int(len(all_episodes)),
        "n_analysis_first_episode": int(len(first_episode)),
        "n_patients": int(first_episode["anon_id"].nunique()) if len(first_episode) else 0,
        "n_high_adi_first_episode": int((first_episode["high_ADI"] == 1).sum()) if len(first_episode) else 0,
        "n_low_adi_first_episode": int((first_episode["high_ADI"] == 0).sum()) if len(first_episode) else 0,
        "outcome_rate_first_episode": float(first_episode["outcome_nonsusceptible"].mean()) if len(first_episode) else None,
    }
    (OUTPUT_DERIVED / "analysis_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
