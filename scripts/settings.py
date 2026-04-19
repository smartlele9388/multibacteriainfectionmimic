from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\adamk\Desktop\mimic-iv-3.1\armd_mgb_esbl_mediation")
ARMD_ROOT = Path(r"C:\Users\adamk\Desktop\mimic-iv-3.1\antibiotic-resistance-microbiology-dataset-mass-general-brigham-armd-mgb-1.0.0\antibiotic-resistance-microbiology-dataset-mass-general-brigham-armd-mgb-1.0.0")
MC_MED_ROOT = Path(r"C:\Users\adamk\Desktop\mimic-iv-3.1\MC-MED")
OUTPUT_DERIVED = PROJECT_ROOT / "output" / "derived"
OUTPUT_TABLES = PROJECT_ROOT / "output" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "output" / "figures"

TARGET_ORGANISMS = {
    "ESCHERICHIA COLI": "Escherichia coli",
    "KLEBSIELLA PNEUMONIAE": "Klebsiella pneumoniae",
    "KLEBSIELLA OXYTOCA": "Klebsiella oxytoca",
    "KLEBSIELLA AEROGENES": "Klebsiella aerogenes",
    "ENTEROBACTER CLOACAE": "Enterobacter cloacae",
    "CITROBACTER FREUNDII": "Citrobacter freundii",
    "CITROBACTER KOSERI": "Citrobacter koseri",
    "MORGANELLA MORGANII": "Morganella morganii",
    "PROTEUS MIRABILIS": "Proteus mirabilis",
    "PROVIDENCIA SPP.": "Providencia spp.",
    "SERRATIA MARCESCENS": "Serratia marcescens",
}

BROAD_SPECTRUM_CODES = {"CRO", "CAZ", "FEP", "TZP"}
PRIMARY_PHENO_POSITIVE = {"Intermediate", "Resistant"}
RESISTANT_ONLY = {"Resistant"}

ABX_CLASS_GROUPS = {
    "ceph_90": {"cephalosporin", "extended_spectrum_cephalosporin"},
    "fq_90": {"fluoroquinolone"},
    "carb_90": {"carbapenem"},
    "blbli_90": {"beta_lactam_beta_lactamase_inhibitor", "extended_spectrum_penicillin"},
    "tmp_smx_90": {"trimethoprim_sulfamethoxazole", "sulfonamide"},
    "glyco_90": {"glycopeptide"},
    "anti_staph_bl_90": {"anti_staph_beta_lactam"},
}

INVASIVE_PROCEDURES = {
    "cvc": "cvc_30",
    "dialysis": "dialysis_30",
    "urethral_catheter": "foley_30",
    "mechvent": "mechvent_30",
    "surgical_procedure": "surgery_30",
}

CARE_SETTING_MAP = {
    "hosp_ward_IP": "inpatient",
    "hosp_ward_OP": "outpatient",
    "hosp_ward_ER": "emergency room",
    "hosp_ward_UC": "urgent care",
    "hosp_ward_day_surg": "day surgery",
}

COMMON_USECOLS = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]

ML_BASELINE_COVARIATES = [
    "elix_count",
    "age_cat",
    "gender_collapsed",
    "culture_site_group",
    "current_care_setting",
    "organism",
]
