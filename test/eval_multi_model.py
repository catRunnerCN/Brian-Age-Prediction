#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_multi_model.py

This script evaluates two brain age prediction models using separate test CSV files.
For each MRI sample, both models produce predictions, and then a weighted average is computed.
The weight for Model1 (w1) is varied from 17.464 to 34.464 (increment by 1), while the weight for Model2 (w2)
remains fixed at 3.2844. The combined prediction is compared with the true age to compute the MAE for each weight combination.
Finally, a line graph is plotted with the x-axis showing the weight ratio (w1:w2) and the y-axis showing the combined test MAE.
"""

import os
import numpy as np
import pandas as pd
import skimage.transform
import matplotlib.pyplot as plt
import tensorflow as tf


def mri_data_generator(df, batch_size=2, do_resize=False, new_shape=(64, 64, 64), shuffle=False, infinite=False):
    index = 0
    n = len(df)
    while True:
        if shuffle and index == 0:
            df = df.sample(frac=1).reset_index(drop=True)
        X_batch = []
        y_batch = []
        for _ in range(batch_size):
            if index >= n:
                index = 0
                if not infinite:
                    break
            row = df.iloc[index]
            file_path = row["file_path"]
            age = row["age"]
            data = np.load(file_path, mmap_mode='r')
            vol = data[0, 0, ...].astype(np.float32)
            if do_resize:
                vol = skimage.transform.resize(vol, new_shape, order=1, preserve_range=True).astype(np.float32)
            m = vol.mean()
            s = vol.std() + 1e-8
            vol = (vol - m) / s
            vol = np.expand_dims(vol, axis=-1)
            X_batch.append(vol)
            y_batch.append(age)
            index += 1
        if len(X_batch) == 0:
            break
        X_batch = np.stack(X_batch, axis=0)
        y_batch = np.array(y_batch, dtype=np.float32)
        yield X_batch, y_batch
        if not infinite and index >= n:
            break


def main():
    csv_path_model1 = "val_index.csv"  # CSV file for Model1
    csv_path_model2 = "val_quasiraw_index.csv"  # CSV file for Model2

    df_test1 = pd.read_csv(csv_path_model1)
    df_test2 = pd.read_csv(csv_path_model2)

    if len(df_test1) != len(df_test2):
        raise ValueError("The two test CSV files do not have the same number of samples!")
    num_samples = len(df_test1)
    print("Total test samples:", num_samples)

    batch_size = 4
    steps = num_samples // batch_size

    test_gen1 = mri_data_generator(df_test1, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)
    test_gen2 = mri_data_generator(df_test2, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)

    model_dir1 = r"F:\Brain_Age_Estimation\brain_age_estimation\code\train\model_earlystopping_p=3_2025_4"  # Model1 with MAE = 3.2844
    model_dir2 = r"F:\Brain_Age_Estimation\brain_age_estimation\code\train\quasiraw_model_earlystopping_p=3_2025_4"  # Model2 with MAE = 4.3660

    model1 = tf.keras.models.load_model(model_dir1)
    model2 = tf.keras.models.load_model(model_dir2)
    print("Model 1 loaded from", model_dir1)
    print("Model 2 loaded from", model_dir2)

    fixed_weight2 = 3.2844
    weight1_values = np.arange(3.2844, 34.2844 + 1e-6, 1.0)

    mae_results = []
    weight_ratios = []

    for weight1 in weight1_values:
        test_gen1 = mri_data_generator(df_test1, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)
        test_gen2 = mri_data_generator(df_test2, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)

        combined_predictions = []
        true_ages_list = []

        for _ in range(steps):
            X_batch1, y_batch1 = next(test_gen1)
            X_batch2, y_batch2 = next(test_gen2)
            if not np.allclose(y_batch1, y_batch2):
                raise ValueError("Mismatch between ages in the two test CSV files!")
            preds1 = model1.predict(X_batch1)
            preds2 = model2.predict(X_batch2)
            combined = (weight1 * preds1 + fixed_weight2 * preds2) / (weight1 + fixed_weight2)
            combined_predictions.append(combined)
            true_ages_list.append(y_batch1)

        combined_predictions = np.vstack(combined_predictions)
        true_ages_arr = np.concatenate(true_ages_list)
        current_mae = np.mean(np.abs(combined_predictions.flatten() - true_ages_arr))
        mae_results.append(current_mae)

        ratio_str = f"{weight1}:{fixed_weight2}"
        weight_ratios.append(ratio_str)
        print(f"Weight ratio {ratio_str}, Combined Test MAE: {current_mae:.4f}")

    x_values = [w1 / fixed_weight2 for w1 in weight1_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, mae_results, marker='o', linestyle='-', color='b')
    plt.xlabel("Weight1/Weight2 Ratio", fontsize=14)
    plt.ylabel("Combined Test MAE", fontsize=14)
    plt.title("Effect of Weighting on Combined Model Test MAE", fontsize=16)
    plt.xticks(x_values, weight_ratios, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("weight_mae_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
