#!/usr/bin/env python3
"""
Merge all evaluation datasets from /Datasets/eval into a single lerobot dataset.
Preserves original data in /Datasets/eval.
"""

import sys
from pathlib import Path
import json

# Add scripts to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from utils.dataset_processing import merge_datasets


def ensure_tasks_parquet(dataset_dir: Path) -> bool:
    """
    Check if a dataset has tasks.parquet. If not, create a minimal one.
    Returns True if dataset is valid, False if it's incomplete and should be skipped.
    """
    meta_dir = dataset_dir / "meta"
    tasks_path = meta_dir / "tasks.parquet"
    
    if tasks_path.exists():
        return True
    
    # Check if at least data and info.json exist
    if not (dataset_dir / "data").exists() or not (meta_dir / "info.json").exists():
        return False
    
    # Try to create minimal tasks.parquet
    try:
        import pandas as pd
        # Create minimal tasks dataframe
        tasks_df = pd.DataFrame({
            "task_index": [0],
            "task": ["unknown"],
            "episode_index": [0]
        })
        tasks_df.to_parquet(tasks_path, index=False)
        print(f"  ⚠️  Created minimal tasks.parquet for {dataset_dir.name}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create tasks.parquet for {dataset_dir.name}: {e}")
        return False


def main():
    eval_dir = Path(__file__).parent / "Datasets" / "eval"
    output_dir = Path(__file__).parent / "Datasets" / "eval_merged"
    
    # Find all evaluation episode directories (001, 002, ..., 136, etc.)
    all_dirs = sorted([
        d for d in eval_dir.iterdir() 
        if d.is_dir() and d.name.isdigit()
    ])
    
    if not all_dirs:
        print(f"❌ No episode directories found in {eval_dir}")
        sys.exit(1)
    
    print(f"📦 Found {len(all_dirs)} episode directories")
    print(f"🔗 Merging from: {eval_dir}")
    print(f"📁 Output to: {output_dir}")
    
    # Validate and prepare datasets
    print("\n🔍 Checking dataset integrity...")
    valid_dirs = []
    skipped = []
    
    for dataset_dir in all_dirs:
        if ensure_tasks_parquet(dataset_dir):
            valid_dirs.append(dataset_dir)
        else:
            skipped.append(dataset_dir.name)
    
    if skipped:
        print(f"⏭️  Skipped {len(skipped)} incomplete datasets: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}")
    
    if not valid_dirs:
        print("❌ No valid datasets to merge!")
        sys.exit(1)
    
    print(f"✓ Valid datasets: {len(valid_dirs)}")
    
    # Ensure output directory doesn't exist (lerobot requires it to not exist)
    import shutil
    if output_dir.exists():
        print(f"\n⚠️  Removing existing output directory...")
        shutil.rmtree(output_dir)
        import time
        time.sleep(0.5)  # Brief pause to ensure removal is complete
    
    # Merge datasets
    print(f"\n🔗 Merging {len(valid_dirs)} datasets...")
    try:
        merge_datasets(
            source_roots=valid_dirs,
            output_root=output_dir,
            output_repo_id="eval_merged",
            merge_custom_meta=True,
        )
        print(f"\n✅ Successfully merged {len(valid_dirs)} datasets!")
        print(f"📁 Output location: {output_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Merge failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
