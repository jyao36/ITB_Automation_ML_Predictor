"""
Simple prediction script for ITB_Automation_ML_Predictor.

This script:
- Merges pVACtools-style MHC class I and class II outputs for a single case
- Applies the published imputer and label encoders
- Uses the trained Balanced Random Forest model to score each epitope
- Writes a single aggregated TSV with updated ML predictions for MHC class I

Inputs (from data/predict_new_case_data):
- <sample>.MHC_I.all_epitopes.aggregated.tsv
- <sample>.MHC_I.all_epitopes.tsv
- <sample>.MHC_II.all_epitopes.aggregated.tsv

Usage (from repository root):
    python scripts/predict.py \\
        --sample-name Test \\
        --artifacts-dir model_artifacts \\
        --output-dir results/predictions

This implementation is adapted from pvactools' ml_predictor module, but
restricted to the ITB_Automation_ML_Predictor repository layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import pickle


def _get_repo_paths() -> Tuple[Path, Path, Path]:
    """Return base, data, and default artifacts directories for this repo."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "predict_new_case_data"
    artifacts_dir = base_dir / "model_artifacts"
    return base_dir, data_dir, artifacts_dir


def _resolve_artifact_paths(
    artifacts_dir: Path,
    artifacts_version: str,
) -> Tuple[Path, Path, Path]:
    """
    Resolve paths to model artifacts in the given directory for a specific version.

    For artifacts_version='numpy126', expected layout is:
      <artifacts_dir>/numpy126/
        - rf_downsample_model_numpy126.pkl
        - trained_imputer_numpy126.joblib
        - label_encoders_numpy126.pkl
    """
    artifacts_subdir = artifacts_dir / artifacts_version
    model_path = artifacts_subdir / f"rf_downsample_model_{artifacts_version}.pkl"
    imputer_path = artifacts_subdir / f"trained_imputer_{artifacts_version}.joblib"
    encoders_path = artifacts_subdir / f"label_encoders_{artifacts_version}.pkl"
    return model_path, imputer_path, encoders_path


def merge_and_prepare_data(
    class1_aggregated_path: Path,
    class1_all_epitopes_path: Path,
    class2_aggregated_path: Path,
) -> pd.DataFrame:
    """
    Merge class I / class II files and engineer features for ML prediction.

    This is a trimmed version of pvactools' merge_and_prepare_data,
    but preserves the model's expected feature space.
    """
    print("Reading input files...")

    mhc1_agg_columns = [
        "ID",
        "Index",
        "Best Peptide",
        "IC50 MT",
        "IC50 WT",
        "%ile MT",
        "%ile WT",
        "Allele",
        "Allele Expr",
        "DNA VAF",
        "TSL",
        "Num Passing Peptides",
        "Pos",
        "Prob Pos",
        "RNA Depth",
        "RNA Expr",
        "RNA VAF",
        "Ref Match",
        "Evaluation",
    ]
    mhc2_agg_columns = [
        "ID",
        "%ile MT",
        "%ile WT",
        "IC50 MT",
        "IC50 WT",
    ]
    mhc1_allepi_columns = [
        "Index",
        "MT Epitope Seq",
        "HLA Allele",
        "Gene of Interest",
        "Corresponding Fold Change",
        "Variant Type",
        "Best MT IC50 Score",
        "Best MT Percentile",
        "Corresponding WT IC50 Score",
        "Corresponding WT Percentile",
        "Biotype",
        "cysteine_count",
    ]
    predictor_columns = [
        "MHCflurry MT IC50 Score",
        "MHCflurry MT Percentile",
        "MHCflurry WT IC50 Score",
        "MHCflurry WT Percentile",
        "MHCflurryEL Presentation MT Percentile",
        "MHCflurryEL Presentation MT Score",
        "MHCflurryEL Presentation WT Percentile",
        "MHCflurryEL Presentation WT Score",
        "MHCflurryEL Processing MT Score",
        "MHCflurryEL Processing WT Score",
        "MHCnuggetsI MT IC50 Score",
        "MHCnuggetsI MT Percentile",
        "MHCnuggetsI WT IC50 Score",
        "MHCnuggetsI WT Percentile",
        "Median Fold Change",
        "NetMHC MT IC50 Score",
        "NetMHC MT Percentile",
        "NetMHC WT IC50 Score",
        "NetMHC WT Percentile",
        "NetMHCcons MT IC50 Score",
        "NetMHCcons MT Percentile",
        "NetMHCcons WT IC50 Score",
        "NetMHCcons WT Percentile",
        "NetMHCpan MT IC50 Score",
        "NetMHCpan MT Percentile",
        "NetMHCpan WT IC50 Score",
        "NetMHCpan WT Percentile",
        "NetMHCpanEL MT Presentation Score",
        "NetMHCpanEL MT Percentile",
        "NetMHCpanEL WT Presentation Score",
        "NetMHCpanEL WT Percentile",
        "Peptide Length",
        "PickPocket MT IC50 Score",
        "PickPocket MT Percentile",
        "PickPocket WT IC50 Score",
        "PickPocket WT Percentile",
        "SMM MT IC50 Score",
        "SMM MT Percentile",
        "SMM WT IC50 Score",
        "SMM WT Percentile",
        "SMMPMBEC MT IC50 Score",
        "SMMPMBEC MT Percentile",
        "SMMPMBEC WT IC50 Score",
        "SMMPMBEC WT Percentile",
    ]

    def get_column_info(file_path: Path, requested_columns):
        available_cols = pd.read_csv(file_path, sep="\t", nrows=0).columns.tolist()
        filtered_cols = [col for col in requested_columns if col in available_cols]
        missing_cols = [col for col in requested_columns if col not in available_cols]
        if missing_cols:
            print(
                f"Caution: missing columns in {file_path.name} will be filled with NA: {missing_cols}"
            )
        return filtered_cols, missing_cols

    def add_missing_columns(df: pd.DataFrame, missing_cols) -> pd.DataFrame:
        for col in missing_cols:
            df[col] = np.nan
        return df

    mhc1_agg_cols_available, mhc1_agg_cols_missing = get_column_info(
        class1_aggregated_path, mhc1_agg_columns
    )
    mhc1_agg_df = pd.read_csv(
        class1_aggregated_path,
        sep="\t",
        na_values=["NA", "NaN", ""],
        keep_default_na=False,
        usecols=mhc1_agg_cols_available,
    )
    mhc1_agg_df = add_missing_columns(mhc1_agg_df, mhc1_agg_cols_missing)

    mhc1_allepi_cols_available, mhc1_allepi_cols_missing = get_column_info(
        class1_all_epitopes_path, mhc1_allepi_columns + predictor_columns
    )
    mhc1_allepi_df = pd.read_csv(
        class1_all_epitopes_path,
        sep="\t",
        na_values=["NA", "NaN", ""],
        keep_default_na=False,
        usecols=mhc1_allepi_cols_available,
    )
    mhc1_allepi_df = add_missing_columns(mhc1_allepi_df, mhc1_allepi_cols_missing)

    mhc2_agg_cols_available, mhc2_agg_cols_missing = get_column_info(
        class2_aggregated_path, mhc2_agg_columns
    )
    mhc2_agg_df = pd.read_csv(
        class2_aggregated_path,
        sep="\t",
        na_values=["NA", "NaN", ""],
        keep_default_na=False,
        usecols=mhc2_agg_cols_available,
    )
    mhc2_agg_df = add_missing_columns(mhc2_agg_df, mhc2_agg_cols_missing)

    mhc1_agg_df.rename(
        columns={
            "IC50 MT": "IC50 MT class1",
            "IC50 WT": "IC50 WT class1",
            "%ile MT": "%ile MT class1",
            "%ile WT": "%ile WT class1",
        },
        inplace=True,
    )
    mhc2_agg_df.rename(
        columns={
            "IC50 MT": "IC50 MT class2",
            "IC50 WT": "IC50 WT class2",
            "%ile MT": "%ile MT class2",
            "%ile WT": "%ile WT class2",
        },
        inplace=True,
    )

    if mhc1_agg_df.shape[0] != mhc2_agg_df.shape[0]:
        print(
            "Warning: Class I and Class II aggregated files have different row counts; "
            "this may cause NA predictions for some rows."
        )

    merged_df = pd.merge(mhc1_agg_df, mhc2_agg_df, on="ID")
    merged_all = pd.merge(
        merged_df,
        mhc1_allepi_df,
        how="inner",
        left_on=["Index", "Best Peptide", "Allele"],
        right_on=["Index", "MT Epitope Seq", "HLA Allele"],
    )

    # Pos parsing
    if merged_all["Pos"].dtype == "object":
        pos_extracted = merged_all["Pos"].astype(str).str.extract(r"^(\d+)")
        merged_all["Pos"] = pos_extracted[0].astype("Int64").astype(float)
    else:
        merged_all["Pos"] = merged_all["Pos"].astype(float)

    # Prob Pos parsing
    merged_all["Prob Pos"] = (
        merged_all["Prob Pos"]
        .fillna("0")
        .astype(str)
        .str.split(",")
        .apply(
            lambda x: int(float(x[0])) if x[0].replace(".", "", 1).isdigit() else 0
        )
    )

    merged_all["Prob match"] = merged_all.apply(
        lambda row: "True"
        if pd.notna(row["Pos"])
        and row["Pos"]
        in [
            int(x)
            for x in str(row["Prob Pos"]).split(",")
            if x.replace(".", "", 1).isdigit()
        ]
        else "False",
        axis=1,
    )
    merged_all["Prob match"] = merged_all["Prob match"].map(
        {"True": True, "False": False}
    ).astype(bool)

    merged_all["Gene of Interest"] = (
        merged_all["Gene of Interest"]
        .fillna("False")
        .map({"True": True, "False": False})
        .astype(bool)
    )

    def convert_tsl_to_int(value):
        if pd.isna(value):
            return 6
        if isinstance(value, str) and value.lower() in ["not supported", "na", "n/a", ""]:
            return 6
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 6

    merged_all["TSL"] = merged_all["TSL"].apply(convert_tsl_to_int).astype(int)

    rename_map_v7 = {
        "NetMHCpanEL MT Presentation Score": "NetMHCpanEL MT IC50 Score",
        "NetMHCpanEL WT Presentation Score": "NetMHCpanEL WT IC50 Score",
    }
    merged_all = merged_all.rename(columns=rename_map_v7)

    merged_all = merged_all.drop(
        columns=["Index", "MT Epitope Seq", "HLA Allele", "Best Peptide", "Allele"]
    )

    return merged_all


def clean_and_impute_data(
    merged_all: pd.DataFrame,
    artifacts_dir: Path,
    artifacts_version: str,
) -> pd.DataFrame:
    """Apply label encoders and imputer from the saved artifacts."""
    print("Loading imputer and label encoders...")
    _model_path, imputer_path, encoders_path = _resolve_artifact_paths(
        artifacts_dir, artifacts_version
    )

    imputer = joblib.load(imputer_path)
    with open(encoders_path, "rb") as f:
        label_encoders = pickle.load(f)

    exclude_columns = ["ID", "Evaluation"]
    columns_to_impute = merged_all.columns.difference(exclude_columns)

    excluded_data = merged_all[exclude_columns].copy()
    data_to_impute = merged_all[columns_to_impute].copy()

    categorical_columns = data_to_impute.select_dtypes(include=["category", "object"]).columns
    for col in categorical_columns:
        if col in label_encoders:
            le = label_encoders[col]
            data_to_impute.loc[:, col] = data_to_impute[col].map(
                lambda x: le.transform([x])[0]
                if x in le.classes_
                else le.transform([le.classes_[0]])[0]
            )

    imputed_data = imputer.transform(data_to_impute)
    imputed_data = pd.DataFrame(imputed_data, columns=columns_to_impute)

    post_imputed_data = pd.concat(
        [excluded_data.reset_index(drop=True), imputed_data.reset_index(drop=True)],
        axis=1,
    )
    return post_imputed_data


def make_ml_predictions(
    post_imputed_data: pd.DataFrame,
    artifacts_dir: Path,
    artifacts_version: str,
    ml_threshold_accept: float = 0.55,
    ml_threshold_reject: float = 0.30,
) -> pd.DataFrame:
    """Load RF model and add prediction columns."""
    print("Loading model and making predictions...")
    model_path, _imputer_path, _encoders_path = _resolve_artifact_paths(
        artifacts_dir, artifacts_version
    )
    rf_model = joblib.load(model_path)

    predict_cols = [
        col
        for col in post_imputed_data.columns
        if col not in ["ID", "Evaluation", "patient_id"]
    ]

    probs = rf_model.predict_proba(post_imputed_data[predict_cols])[:, 1]
    post_imputed_data["Accept_pred_prob"] = probs

    post_imputed_data["Evaluation_pred"] = np.where(
        post_imputed_data["Accept_pred_prob"].isna(),
        "Pending",
        np.where(
            post_imputed_data["Accept_pred_prob"] >= ml_threshold_accept,
            "Accept",
            np.where(
                post_imputed_data["Accept_pred_prob"] > ml_threshold_reject,
                "Review",
                "Reject",
            ),
        ),
    )

    return post_imputed_data


def create_final_output(
    post_imputed_data: pd.DataFrame,
    original_agg_file_path: Path,
    output_dir: Path,
    sample_name: str,
) -> Path:
    """Merge predictions back into the original class I aggregated file."""
    print("Writing final prediction file...")
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_df = pd.read_csv(
        original_agg_file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    original_columns = list(orig_df.columns)
    cols_to_preserve = [c for c in original_columns if c != "Evaluation"]

    base_df = orig_df.drop(columns=["Evaluation"])
    merged_df = base_df.merge(
        post_imputed_data[["ID", "Evaluation_pred", "Accept_pred_prob"]],
        on="ID",
        how="left",
    ).rename(columns={"Evaluation_pred": "Evaluation"})
    merged_df["Evaluation"] = merged_df["Evaluation"].fillna("Pending")

    merged_df["ML Prediction (score)"] = merged_df.apply(
        lambda row: "NA"
        if (
            pd.isna(row["Evaluation"])
            or row["Evaluation"] == "Pending"
            or pd.isna(row["Accept_pred_prob"])
        )
        else f"{row['Evaluation']} ({round(row['Accept_pred_prob'], 2)})",
        axis=1,
    )
    merged_df.loc[merged_df["Evaluation"] == "Review", "Evaluation"] = "Pending"
    final_df = merged_df.drop(columns=["Accept_pred_prob"])

    final_df[cols_to_preserve] = orig_df[cols_to_preserve]

    output_file = output_dir / f"{sample_name}.MHC_I.all_epitopes.aggregated.ML_predict.tsv"
    final_df.to_csv(output_file, float_format="%.3f", sep="\t", index=False, na_rep="NA")
    print(f"ML predictions saved to: {output_file}")
    return output_file


def run_predictions(
    sample_name: str,
    artifacts_dir: Path | None = None,
    artifacts_version: str = "numpy126",
    output_dir: Path | None = None,
    ml_threshold_accept: float = 0.55,
    ml_threshold_reject: float = 0.30,
) -> Path:
    """End-to-end prediction pipeline for a single case."""
    if ml_threshold_reject > ml_threshold_accept:
        raise ValueError(
            f"ml_threshold_reject ({ml_threshold_reject}) must be <= ml_threshold_accept ({ml_threshold_accept})."
        )

    _, data_dir, default_artifacts_dir = _get_repo_paths()
    artifacts_dir = artifacts_dir or default_artifacts_dir
    artifacts_dir = Path(artifacts_dir)

    if output_dir is None:
        output_dir = Path("results") / "predictions"
    output_dir = Path(output_dir)

    class1_agg = data_dir / f"{sample_name}.MHC_I.all_epitopes.aggregated.tsv"
    class1_all = data_dir / f"{sample_name}.MHC_I.all_epitopes.tsv"
    class2_agg = data_dir / f"{sample_name}.MHC_II.all_epitopes.aggregated.tsv"

    if not class1_agg.exists():
        raise FileNotFoundError(f"Class I aggregated file not found: {class1_agg}")
    if not class1_all.exists():
        raise FileNotFoundError(f"Class I all epitopes file not found: {class1_all}")
    if not class2_agg.exists():
        raise FileNotFoundError(f"Class II aggregated file not found: {class2_agg}")

    print(f"Starting prediction pipeline using artifacts version '{artifacts_version}'...")
    merged_all = merge_and_prepare_data(class1_agg, class1_all, class2_agg)
    post_imputed = clean_and_impute_data(merged_all, artifacts_dir, artifacts_version)
    post_imputed = make_ml_predictions(
        post_imputed,
        artifacts_dir,
        artifacts_version,
        ml_threshold_accept=ml_threshold_accept,
        ml_threshold_reject=ml_threshold_reject,
    )
    output_file = create_final_output(post_imputed, class1_agg, output_dir, sample_name)
    print("Prediction pipeline completed successfully.")
    return output_file


# Hard-coded configuration for manuscript reproducibility
SAMPLE_NAME = "Test"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "model_artifacts"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions"
ARTIFACTS_VERSION = "numpy126"
ML_THRESHOLD_ACCEPT = 0.55
ML_THRESHOLD_REJECT = 0.30


if __name__ == "__main__":
    run_predictions(
        sample_name=SAMPLE_NAME,
        artifacts_dir=ARTIFACTS_DIR,
        artifacts_version=ARTIFACTS_VERSION,
        output_dir=OUTPUT_DIR,
        ml_threshold_accept=ML_THRESHOLD_ACCEPT,
        ml_threshold_reject=ML_THRESHOLD_REJECT,
    )

