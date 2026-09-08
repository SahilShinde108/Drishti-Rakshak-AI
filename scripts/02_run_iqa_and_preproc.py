import os
import cv2
import json
import logging
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.iqa.feedback_generator import run_full_iqa
from src.preprocessing.border_cropper import RetinalBorderCropper
from src.preprocessing.sequential_pipeline import SequentialPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_image(base_dir: Path, img_id: str):
    search_paths = [
        base_dir / "data" / "Raw" / "B. Disease Grading" / "B. Disease Grading" / "1. Original Images" / "a. Training Set" / f"{img_id}.jpg",
        base_dir / "data" / "Raw" / "B. Disease Grading" / "B. Disease Grading" / "1. Original Images" / "b. Testing Set" / f"{img_id}.jpg",
        base_dir / "data" / "Raw" / "A. Segmentation" / "A. Segmentation" / "1. Original Images" / "a. Training Set" / f"{img_id}.jpg",
        base_dir / "data" / "Raw" / "A. Segmentation" / "A. Segmentation" / "1. Original Images" / "b. Testing Set" / f"{img_id}.jpg",
        base_dir / "data" / "Raw" / "messidor-2" / "preprocess" / f"{img_id}_PP.png",
        base_dir / "data" / "Raw" / "messidor-2" / "preprocess" / f"{img_id}.png",
        base_dir / "data" / "Preprocessed_Images" / f"{img_id}_preprocessed.jpg",
        base_dir / "data" / "Preprocessed_Images" / f"{img_id}.png",
        base_dir / "data" / "Preprocessed_Images" / f"{img_id}.jpg",
        base_dir / "data" / "raw" / f"{img_id}.jpg",
        base_dir / "data" / "raw" / f"{img_id}.png",
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None

def run_pipeline(demo_limit=None, target_split=None):
    base_dir = Path(__file__).resolve().parents[1]
    splits_dir = base_dir / "data" / "splits"
    out_dir = base_dir / "data" / "processed"
    outputs_dir = base_dir / "outputs"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cropper = RetinalBorderCropper()
    preprocessor = SequentialPreprocessor()
    
    splits_to_process = ["train", "val", "test"] if not target_split else [target_split]
    
    all_iqa_results = []
    summary = {"total_processed": 0, "gradable": 0, "ungradable": 0, "errors": 0}

    for split in splits_to_process:
        manifest_path = splits_dir / f"{split}_manifest.csv"
        if not manifest_path.exists():
            logger.warning(f"Manifest {manifest_path} not found. Skipping {split}.")
            continue
            
        df = pd.read_csv(manifest_path)
        if demo_limit:
            df = df.head(demo_limit)
            
        logger.info(f"Processing {split} split ({len(df)} images)...")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_id = str(row['image_id'])
            img_path = find_image(base_dir, img_id)
            
            if img_path is None or not img_path.exists():
                logger.error(f"Image missing for image_id: {img_id}")
                summary["errors"] += 1
                continue
                
            image = cv2.imread(str(img_path))
            
            # 1. IQA
            iqa = run_full_iqa(image)
            iqa["image_id"] = img_id
            iqa["split"] = split
            all_iqa_results.append(iqa)
            
            summary["total_processed"] += 1
            
            # 2. Preprocess if gradable
            if iqa["is_gradable"]:
                summary["gradable"] += 1
                try:
                    cropped = cropper.crop_and_resize(image)
                    processed = preprocessor.process(cropped)
                    
                    # Save
                    save_path = out_dir / f"{img_id}_processed.jpg"
                    # Denormalize to save
                    to_save = (processed * 255).astype('uint8')
                    cv2.imwrite(str(save_path), to_save)
                    
                except Exception as e:
                    logger.error(f"Error preprocessing {img_id}: {e}")
                    summary["errors"] += 1
            else:
                summary["ungradable"] += 1
                logger.warning(f"Image {img_id} ungradable: {iqa['quality_tier']}")

    # Save outputs
    if all_iqa_results:
        iqa_df = pd.DataFrame(all_iqa_results)
        iqa_df.to_csv(outputs_dir / "iqa_results.csv", index=False)
        
    with open(outputs_dir / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info("Batch processing completed.")
    logger.info(f"Summary: {summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=int, help="Limit number of images to process per split")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Specific split to process")
    args = parser.parse_args()
    
    run_pipeline(args.demo, args.split)
