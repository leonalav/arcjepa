import os
import sys

# CRITICAL: Force CPU backend so libtpu doesn't crash during the multiprocessing spawn
os.environ["PJRT_DEVICE"] = "CPU"

from pathlib import Path
import torch
import numpy as np
from datasets import Dataset

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ARCTrajectoryDataset

def pack_and_upload(data_dir: str, hf_repo_id: str, split: str):
    print(f"\n🚀 Processing {split.upper()} split from {data_dir}...")
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Skipping...")
        return

    recording_files = [str(f) for f in Path(data_dir).glob("*.jsonl")]
    if not recording_files:
        print(f"No .jsonl files found in {data_dir}. Skipping...")
        return
        
    print(f"Found {len(recording_files)} files. Firing up 96-CPU parallel parser...")

    # Load via our optimized class
    dataset = ARCTrajectoryDataset(
        recording_files=recording_files,
        window_size=64,
        stride=16, 
        max_grid_size=64,
        filter_noops=True
    )
    
    print(f"✅ Parsed {len(dataset)} valid chunks. Packing into Arrow format...")
    
    # We will pack the EXACT tensors that your train.py expects
    packed_data = {
        "states": [],
        "actions": [],
        "coords_x": [],
        "coords_y": [],
        "target_states": [],
        "final_state": []
    }
    
    # Iterate using __getitem__ to trigger the proper tensor slicing
    for i in range(len(dataset)):
        item = dataset[i]
        packed_data["states"].append(item['states'].numpy().astype(np.uint8))
        packed_data["actions"].append(item['actions'].numpy().astype(np.uint8))
        packed_data["coords_x"].append(item['coords_x'].numpy().astype(np.uint8))
        packed_data["coords_y"].append(item['coords_y'].numpy().astype(np.uint8))
        packed_data["target_states"].append(item['target_states'].numpy().astype(np.uint8))
        packed_data["final_state"].append(item['final_state'].numpy().astype(np.uint8))

    print(f"📦 Converting to Hugging Face Dataset...")
    hf_dataset = Dataset.from_dict(packed_data)
    
    print(f"☁️ Pushing to {hf_repo_id} (Split: {split})...")
    hf_dataset.push_to_hub(hf_repo_id, split=split)
    print(f"🎉 {split.capitalize()} split successfully uploaded!")

if __name__ == "__main__":
    REPO_ID = "leonidas123/arc3data"
    
    train_dir = str(PROJECT_ROOT / "data" / "recordings" / "train")
    val_dir = str(PROJECT_ROOT / "data" / "recordings" / "val")
    
    pack_and_upload(train_dir, REPO_ID, split="train")
    pack_and_upload(val_dir, REPO_ID, split="validation")