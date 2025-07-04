"""
This file reads MRI files from a .csv file.
It outputs statistics of each MRI in terminal.
It outputs slices of each MRI.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_mri_slices(file_path, pause_time=2.0):
    try:
        data = np.load(file_path, mmap_mode='r')
    except Exception as e:
        print(f"File to load {file_path} ：{e}")
        return

    if data.ndim < 5:
        print(f"File {file_path} has a data shape that is not valid : {data.shape}")
        return

    volume = data[0, 0, ...].astype(np.float32)
    if volume.size == 0:
        print(f"File {file_path} is empty.")
        return

    d, h, w = volume.shape
    x_mid = d // 2
    y_mid = h // 2
    z_mid = w // 2

    slice_x = volume[x_mid, :, :]
    slice_y = volume[:, y_mid, :]
    slice_z = volume[:, :, z_mid]

    def get_stats(slice_array):
        return {
            "mean": np.mean(slice_array),
            "std": np.std(slice_array),
            "min": np.min(slice_array),
            "max": np.max(slice_array)
        }
    stats_x = get_stats(slice_x)
    stats_y = get_stats(slice_y)
    stats_z = get_stats(slice_z)

    print(f"File: {file_path}")
    print(f"  Sagittal (x={x_mid}): mean={stats_x['mean']:.4f}, std={stats_x['std']:.4f}, min={stats_x['min']:.4f}, max={stats_x['max']:.4f}")
    print(f"  Coronal (y={y_mid}): mean={stats_y['mean']:.4f}, std={stats_y['std']:.4f}, min={stats_y['min']:.4f}, max={stats_y['max']:.4f}")
    print(f"  Axial (z={z_mid}): mean={stats_z['mean']:.4f}, std={stats_z['std']:.4f}, min={stats_z['min']:.4f}, max={stats_z['max']:.4f}")

    base_name = os.path.basename(file_path)
    pid = base_name.split('_')[0] if '_' in base_name else base_name

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(slice_x, cmap='gray')
    plt.title(f"{pid}\nSagittal (x={x_mid})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(slice_y, cmap='gray')
    plt.title(f"Coronal (y={y_mid})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(slice_z, cmap='gray')
    plt.title(f"Axial (z={z_mid})")
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(pause_time)
    plt.close()

def main():
    csv_file = r"F:\Brain_Age_Estimation\brain_age_estimation\code\dataset_proc\suspect_samples.csv"
    df = pd.read_csv(csv_file)
    print(f"{len(df)} samples read from {csv_file}")

    for idx, row in df.iterrows():
        file_path = row["file_path"]
        print(f"Show sample {idx+1}/{len(df)}:")
        display_mri_slices(file_path)

if __name__ == "__main__":
    main()
