#!/usr/bin/env python3
"""SignBridge Dataset Preparation Pipeline.

Prepares the 31-category custom webcam dataset by:
1. Validating image integrity across all categories.
2. Inpainting and removing burned-in recording text overlays in the top-left corner
   using OpenCV Telea inpainting without affecting hand gestures or facial features.
3. Sorting each category chronologically to prevent temporal autocorrelation leakage.
4. Splitting chronologically into 70% train (42/class), 15% val (9/class), 15% test (9/class).
5. Saving cleaned images to dataset/cleaned_images and exporting dataset_splits.json
   and dataset_report.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_INPUT_DIR = "/Users/aryansoni/Desktop/SignBridgeDataset/CollectedImages"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "cleaned_images")
DEFAULT_SPLITS_PATH = os.path.join(PROJECT_ROOT, "models", "dataset_splits.json")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "dataset", "dataset_report.json")


def clean_image_overlay(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """Detects and inpaints green and cyan text overlays in the top-left region.

    The collection script rendered:
      - 'Sign: <label>' in green (0, 255, 0) at (20, 40)
      - 'Image: <n>/60' in cyan (255, 255, 0) at (20, 85)

    Confined strictly within y in [0, 115], x in [0, 420].

    Returns:
        Tuple of (cleaned_image, text_pixels_count).
    """
    h, w = img.shape[:2]
    # Restrict search strictly to overlay corner
    crop_h = min(115, h)
    crop_w = min(420, w)
    corner = img[0:crop_h, 0:crop_w]

    # Mask green text: high G, low R, low B
    green_mask = (corner[:, :, 1] > 155) & (corner[:, :, 0] < 85) & (corner[:, :, 2] < 85)
    # Mask cyan text (255, 255, 0 in BGR): high B, high G, low R
    cyan_mask = (corner[:, :, 0] > 155) & (corner[:, :, 1] > 155) & (corner[:, :, 2] < 85)

    text_mask_corner = green_mask | cyan_mask
    pixel_count = int(text_mask_corner.sum())

    if pixel_count == 0:
        return img, 0

    # Dilate slightly to catch anti-aliasing edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_corner = cv2.dilate(text_mask_corner.astype(np.uint8), kernel, iterations=1)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[0:crop_h, 0:crop_w] = dilated_corner

    # Fast inpainting with Telea method
    cleaned = cv2.inpaint(img, full_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned, pixel_count


def process_dataset(
    input_dir: str,
    output_dir: str,
    splits_path: str,
    report_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Prepares and cleans the full image dataset."""
    logger.info("Reading input dataset from: %s", input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    categories = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
    ])
    logger.info("Found %d categories.", len(categories))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    splits_data = {
        "train": [],
        "val": [],
        "test": [],
        "classes": categories,
        "class_to_idx": {cat: idx for idx, cat in enumerate(categories)},
    }

    report_categories = {}
    total_images = 0
    total_cleaned_pixels = 0
    t0 = time.time()

    for cat_idx, cat in enumerate(categories):
        cat_in_dir = os.path.join(input_dir, cat)
        cat_out_dir = os.path.join(output_dir, cat)
        os.makedirs(cat_out_dir, exist_ok=True)

        files = glob.glob(os.path.join(cat_in_dir, "*"))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")) and not os.path.basename(f).startswith(".")]

        # Sort chronologically by modification/creation time to prevent temporal leakage
        files = sorted(files, key=os.path.getmtime)
        n_files = len(files)

        if n_files == 0:
            logger.warning("Category '%s' has 0 images!", cat)
            continue

        n_train = int(round(n_files * train_ratio))
        n_val = int(round(n_files * val_ratio))
        # Ensure remaining go to test
        n_test = n_files - n_train - n_val

        cat_cleaned_pixels = 0
        img_shapes = set()

        for idx, src_path in enumerate(files):
            img = cv2.imread(src_path)
            if img is None:
                logger.error("Corrupted image could not be read: %s", src_path)
                continue

            img_shapes.add(f"{img.shape[1]}x{img.shape[0]}")
            cleaned_img, px_count = clean_image_overlay(img)
            cat_cleaned_pixels += px_count

            dst_filename = os.path.basename(src_path)
            dst_path = os.path.join(cat_out_dir, dst_filename)
            cv2.imwrite(dst_path, cleaned_img)

            # Assign split partition
            rel_path = os.path.join(cat, dst_filename)
            sample_entry = {
                "rel_path": rel_path,
                "category": cat,
                "label": cat_idx,
            }

            if idx < n_train:
                splits_data["train"].append(sample_entry)
            elif idx < n_train + n_val:
                splits_data["val"].append(sample_entry)
            else:
                splits_data["test"].append(sample_entry)

        total_images += n_files
        total_cleaned_pixels += cat_cleaned_pixels
        report_categories[cat] = {
            "total_images": n_files,
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "dimensions": list(img_shapes),
            "text_overlay_removed_pixels": cat_cleaned_pixels,
        }
        logger.info(
            "[%d/%d] Processed '%s': %d images (train=%d, val=%d, test=%d), inpaint_px=%d",
            cat_idx + 1,
            len(categories),
            cat,
            n_files,
            n_train,
            n_val,
            n_test,
            cat_cleaned_pixels,
        )

    # Save splits JSON
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
    logger.info("Saved dataset splits to: %s", splits_path)

    # Save dataset report JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_categories": len(categories),
        "total_images": total_images,
        "split_summary": {
            "train": len(splits_data["train"]),
            "val": len(splits_data["val"]),
            "test": len(splits_data["test"]),
        },
        "total_overlay_pixels_inpainted": total_cleaned_pixels,
        "categories": report_categories,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved dataset report to: %s", report_path)
    logger.info(
        "Dataset preparation completed in %.2fs. Total: %d images (Train: %d, Val: %d, Test: %d).",
        report["elapsed_sec"],
        total_images,
        len(splits_data["train"]),
        len(splits_data["val"]),
        len(splits_data["test"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare and clean sign language dataset.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Path to raw CollectedImages")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Path to save cleaned images")
    parser.add_argument("--splits-path", default=DEFAULT_SPLITS_PATH, help="Path for dataset_splits.json")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help="Path for dataset_report.json")
    args = parser.parse_args()

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits_path=args.splits_path,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
