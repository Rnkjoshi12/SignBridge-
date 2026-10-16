#!/usr/bin/env python3
"""
SignBridge Dataset Inspector
============================

Inspects the sign language dataset, discovers categories, validates images,
checks dimensions, and generates a JSON report.

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --dataset-path "/path/to/dataset"
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing PIL for image validation
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}

# Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATASET = os.path.join(os.path.dirname(PROJECT_ROOT), 'images for phrases')
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_report.json')


def inspect_image(file_path: str) -> dict:
    """Inspect a single image file for validity and properties.

    Args:
        file_path: Path to the image file.

    Returns:
        Dict with 'valid', 'width', 'height', 'format', 'error' keys.
    """
    if not os.path.isfile(file_path):
        return {"valid": False, "width": None, "height": None,
                "format": None, "error": "File does not exist"}

    if os.path.getsize(file_path) == 0:
        return {"valid": False, "width": None, "height": None,
                "format": None, "error": "Empty file"}

    if HAS_PIL:
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                img.load()
                return {
                    "valid": True,
                    "width": img.size[0],
                    "height": img.size[1],
                    "format": img.format or "Unknown",
                    "error": None
                }
        except Exception as e:
            return {"valid": False, "width": None, "height": None,
                    "format": None, "error": str(e)}
    else:
        # Basic check without PIL
        return {"valid": True, "width": None, "height": None,
                "format": os.path.splitext(file_path)[1].upper(), "error": None}


def find_metadata(dataset_path: str) -> list:
    """Find README or metadata files in the dataset directory."""
    metadata_exts = {'.txt', '.md', '.json', '.csv', '.yaml', '.yml'}
    results = []
    for f in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, f)):
            _, ext = os.path.splitext(f.lower())
            if ext in metadata_exts or f.lower().startswith('readme'):
                results.append(f)
    return results


def inspect_dataset(dataset_path: str, report_path: str) -> dict:
    """Run full dataset inspection.

    Args:
        dataset_path: Path to the dataset directory.
        report_path: Path to write the JSON report.

    Returns:
        The inspection report as a dictionary.
    """
    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset not found at: {dataset_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"SignBridge Dataset Inspection")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_path}")
    print(f"PIL available: {HAS_PIL}")

    # Discover categories
    categories = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and not d.startswith('.')
    ])

    print(f"\nFound {len(categories)} categories")
    print(f"{'-'*70}")
    print(f"{'#':>3}  {'Category':<30}  {'Images':>7}  {'Dims':>12}  {'Status':>8}")
    print(f"{'-'*70}")

    category_counts = {}
    category_details = []
    corrupted_files = []
    dimension_counts = {}
    total_images = 0

    for idx, cat in enumerate(categories, 1):
        cat_dir = os.path.join(dataset_path, cat)
        images = sorted([
            f for f in os.listdir(cat_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            and not f.startswith('.')
        ])

        count = len(images)
        total_images += count
        category_counts[cat] = count

        # Check a sample image for dimensions
        dims = "N/A"
        cat_corrupted = 0
        if images:
            sample_path = os.path.join(cat_dir, images[0])
            info = inspect_image(sample_path)
            if info["valid"] and info["width"]:
                dims = f"{info['width']}x{info['height']}"
                dim_key = dims
                dimension_counts[dim_key] = dimension_counts.get(dim_key, 0) + count
            if not info["valid"]:
                cat_corrupted += 1
                corrupted_files.append({
                    "category": cat, "file": images[0], "error": info["error"]
                })

        status = "OK" if cat_corrupted == 0 else f"{cat_corrupted} BAD"
        cat_display = cat if len(cat) <= 30 else cat[:27] + "..."
        print(f"{idx:>3}  {cat_display:<30}  {count:>7}  {dims:>12}  {status:>8}")

        category_details.append({
            "category": cat,
            "image_count": count,
            "corrupted": cat_corrupted
        })

    # Summary
    print(f"{'-'*70}")
    counts = list(category_counts.values())
    min_c, max_c = min(counts), max(counts)
    balance = "Perfect" if min_c == max_c else f"Min={min_c}, Max={max_c}"

    print(f"\nSummary:")
    print(f"  Categories:     {len(categories)}")
    print(f"  Total images:   {total_images}")
    print(f"  Balance:        {balance} ({min_c} images per category)")
    print(f"  Dimensions:     {', '.join(dimension_counts.keys()) or 'Unknown'}")
    print(f"  Corrupted:      {len(corrupted_files)}")

    metadata = find_metadata(dataset_path)
    if metadata:
        print(f"  Metadata files: {', '.join(metadata)}")
    else:
        print(f"  Metadata files: None (sign language not verified)")

    # Build report
    report = {
        "dataset_path": dataset_path,
        "inspected_at": datetime.now().isoformat(),
        "total_categories": len(categories),
        "total_images": total_images,
        "categories": categories,
        "category_counts": category_counts,
        "dimensions": dimension_counts,
        "corrupted_files": corrupted_files,
        "metadata_files": metadata,
        "is_healthy": len(corrupted_files) == 0,
        "notes": (
            "No README or metadata was found to identify the specific "
            "sign language. Categories suggest common conversational phrases. "
            "The sign language identity has not been verified."
        )
    }

    # Write report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    print(f"{'='*70}\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the SignBridge sign language dataset."
    )
    parser.add_argument(
        '--dataset-path', '-d', default=DEFAULT_DATASET,
        help=f"Path to dataset directory (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        '--output-report', '-o', default=DEFAULT_REPORT,
        help=f"Output path for JSON report (default: {DEFAULT_REPORT})"
    )
    args = parser.parse_args()

    inspect_dataset(args.dataset_path, args.output_report)


if __name__ == '__main__':
    main()
