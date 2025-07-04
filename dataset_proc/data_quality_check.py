"""
data_quality_check.py
This file calculate the statistics of each MRI files in file directory.
It outputs a .csv that saves these statistics.
"""

import os
import glob
import numpy as np
import pandas as pd


def compute_mri_stats(file_path):
    try:
        data = np.load(file_path, mmap_mode='r')
    except Exception as e:
        print(f"Fail to load {file_path} : {e}")
        return None

    if data.ndim < 5:
        print(f"File {file_path} has a data shape that is not valid: {data.shape}")
        return None

    volume = data[0, 0, ...].astype(np.float32)
    stats = {
        "mean": float(np.mean(volume)),
        "std": float(np.std(volume)),
        "min": float(np.min(volume)),
        "max": float(np.max(volume))
    }
    return stats


def main():
    folder_path = r"F:\Dataset\train\train_quasiraw"
    file_list = glob.glob(os.path.join(folder_path, "*.npy"))
    print(f"Found {len(file_list)} MRI files in {folder_path}")

    records = []
    for file_path in file_list:
        stats = compute_mri_stats(file_path)
        if stats is not None:
            base_name = os.path.basename(file_path)
            pid = base_name.split('_')[0]  # "sub-100053248969"
            record = {"file_path": file_path, "participant_id": pid}
            record.update(stats)
            records.append(record)

    df_stats = pd.DataFrame(records)
    output_file = "mri_quality_stats_qui.csv"
    df_stats.to_csv(output_file, index=False)
    print(f"Analysis finished. Dealt with {len(df_stats)} files in total, result saved in {output_file}")


if __name__ == "__main__":
    main()
