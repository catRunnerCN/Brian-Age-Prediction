#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_model.py
This file is used to evaluate the model on test set.
It will output test MSE and test MAE, along with a scatter point plot.
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
    csv_path = "val_quasiraw_index.csv"
    df_test = pd.read_csv(csv_path)
    print("Total test samples:", len(df_test))

    batch_size = 8
    test_gen = mri_data_generator(df_test, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)
    steps = len(df_test) // batch_size
    print("Steps for evaluation:", steps)

    model_dir = r"F:\Brain_Age_Estimation\brain_age_estimation\code\train\quasiraw_model_earlystopping_p=3_2025_4"
    model = tf.keras.models.load_model(model_dir)
    print("Model loaded from", model_dir)

    result = model.evaluate(test_gen, steps=steps)
    print("Test MSE: {:.4f}, Test MAE: {:.4f}".format(result[0], result[1]))

    predictions = model.predict(test_gen, steps=steps)
    num_samples = steps * batch_size
    true_ages = df_test.iloc[:num_samples]['age'].values

    plt.figure(figsize=(8, 6))
    plt.scatter(true_ages, predictions.flatten(), color='b', alpha=0.6, label='Predicted Age')

    min_age = min(true_ages.min(), predictions.min())
    max_age = max(true_ages.max(), predictions.max())
    plt.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=2, label='Ideal Prediction')

    plt.xlabel("True Age (years)", fontsize=14)
    plt.ylabel("Predicted Age (years)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Predicted vs. True Age on Test Set", fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig("evaluation_scatter.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
