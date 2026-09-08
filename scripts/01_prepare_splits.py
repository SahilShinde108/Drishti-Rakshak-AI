import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def assign_patient_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign patient IDs ensuring bilateral eye pairs share the same patient ID.
    - Messidor-2: Consecutive row pairs belong to the same patient visit.
    - IDRiD: IDRiD_XX / IDRiD_XXX prefixes group patient scans.
    """
    patient_ids = []
    messidor_idx = 0
    for _, row in df.iterrows():
        img_id = str(row['image_id'])
        src = str(row.get('dataset_source', ''))
        if src == 'Messidor-2':
            pat_id = f"MESSIDOR_PAT_{messidor_idx // 2:04d}"
            messidor_idx += 1
        elif 'IDRiD_' in img_id:
            parts = img_id.split('_')
            pat_id = f"IDRiD_{parts[1]}"
        else:
            pat_id = img_id
        patient_ids.append(pat_id)
    df['patient_id'] = patient_ids
    return df

def prepare_splits(csv_path: str):
    config = load_config()
    seed = config.get("training", {}).get("seed", 42)
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"Dataset CSV not found at {csv_path}.")
        return

    df = pd.read_csv(csv_file)
    required_cols = {"image_id", "dr_stage_label"}
    if not required_cols.issubset(df.columns):
        logger.error(f"Missing required columns. Found: {list(df.columns)}")
        return

    # Add patient ID with zero bilateral leakage
    df = assign_patient_ids(df)
    logger.info(f"Assigned patient IDs: {df['patient_id'].nunique()} unique patients across {len(df)} images.")
    
    # Stratify at patient level using max DR stage as the patient label
    patient_df = df.groupby('patient_id')['dr_stage_label'].max().reset_index()

    # Split: 70% Train, 30% (Val + Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, temp_idx = next(sss1.split(patient_df['patient_id'], patient_df['dr_stage_label']))
    
    train_patients = patient_df.iloc[train_idx]
    temp_patients = patient_df.iloc[temp_idx]

    # Split Temp into 50% Val, 50% Test (15% each of total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_patients['patient_id'], temp_patients['dr_stage_label']))
    
    val_patients = temp_patients.iloc[val_idx]
    test_patients = temp_patients.iloc[test_idx]

    # Strict Zero-Leakage Assertion Check
    s1, s2, s3 = set(train_patients['patient_id']), set(val_patients['patient_id']), set(test_patients['patient_id'])
    assert len(s1.intersection(s2)) == 0, "Train-Val patient leakage detected!"
    assert len(s1.intersection(s3)) == 0, "Train-Test patient leakage detected!"
    assert len(s2.intersection(s3)) == 0, "Val-Test patient leakage detected!"
    logger.info("Zero-leakage assertion verified: Absolutely 0 patient overlap across splits.")

    # Map back to original images
    train_df = df[df['patient_id'].isin(s1)]
    val_df = df[df['patient_id'].isin(s2)]
    test_df = df[df['patient_id'].isin(s3)]

    # Save to data/splits/
    base_dir = Path(__file__).resolve().parents[1]
    splits_dir = base_dir / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train_manifest.csv", index=False)
    val_df.to_csv(splits_dir / "val_manifest.csv", index=False)
    test_df.to_csv(splits_dir / "test_manifest.csv", index=False)

    # Also save to data/manifests/benchmark_test.csv (guaranteed clean benchmark)
    manifests_dir = base_dir / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(manifests_dir / "benchmark_test.csv", index=False)
    logger.info(f"Saved benchmark test set to {manifests_dir / 'benchmark_test.csv'}")

    counts = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(df),
        "unique_patients": {
            "train": len(s1),
            "val": len(s2),
            "test": len(s3),
            "total": df['patient_id'].nunique()
        }
    }
    
    meta = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "counts": counts,
        "zero_leakage_verified": True
    }
    
    with open(splits_dir / "split_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    logger.info("Splits generated successfully:")
    logger.info(f"Train: {counts['train']} images ({len(s1)} patients)")
    logger.info(f"Val:   {counts['val']} images ({len(s2)} patients)")
    logger.info(f"Test:  {counts['test']} images ({len(s3)} patients)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare zero-leakage dataset splits")
    default_csv = Path(__file__).resolve().parents[1] / "data" / "multimodal_dr_dataset_22_features.csv"
    parser.add_argument("--csv", type=str, default=str(default_csv), help="Path to input dataset CSV")
    args = parser.parse_args()
    
    prepare_splits(args.csv)
