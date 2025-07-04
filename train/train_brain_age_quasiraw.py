#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_brain_age_quasiraw.py
The difference between this file and train_brain_age.py is that the input shape is different,
because Quasi-Raw data has a shape of (182,218,182,1)
"""

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import pandas as pd
import skimage.transform
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def mri_data_generator(df, batch_size=2, do_resize=True, new_shape=(64, 64, 64), shuffle=True, infinite=True):
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
            if not infinite:
                break
            else:
                continue
        X_batch = np.stack(X_batch, axis=0)
        y_batch = np.array(y_batch, dtype=np.float32)
        yield X_batch, y_batch

def create_3d_cnn(input_shape):
    model = models.Sequential()
    model.add(layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

class MyEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch + 1}.")


def main():
    index_file = "quasiraw_index.csv"
    df_all = pd.read_csv(index_file)
    print("Total samples in index:", len(df_all))

    train_df, val_df = train_test_split(df_all, test_size=0.2, random_state=42)
    print("Train samples:", len(train_df), "Validation samples:", len(val_df))

    batch_size = 4
    train_gen = mri_data_generator(train_df, batch_size=batch_size, do_resize=False, shuffle=True, infinite=True)
    val_gen = mri_data_generator(val_df, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)

    input_shape = (182, 218, 182, 1)
    model = create_3d_cnn(input_shape)
    model.summary()

    early_stopping = MyEarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    epochs = 200
    steps_per_epoch = len(train_df) // batch_size
    validation_steps = len(val_df) // batch_size

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stopping]
    )

    tf.keras.models.save_model(model, 'quasiraw_model_earlystopping_p=3_2025_4', save_format='tf')
    print("Model saved.")

    # --- F. Plot Training and Validation Curves ---
    epoch_indices = range(1, len(history.history['loss']) + 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    plt.figure(figsize=(10, 4))
    plt.plot(epoch_indices, train_loss, 'bo-', label='Train Loss')
    plt.plot(epoch_indices, val_loss, 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epoch_indices, train_mae, 'bo-', label='Train MAE')
    plt.plot(epoch_indices, val_mae, 'ro-', label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.show()

    result = model.evaluate(
        mri_data_generator(val_df, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False),
        steps=validation_steps
    )
    print(f"Validation MSE: {result[0]:.4f}, Validation MAE: {result[1]:.4f}")

    val_gen_for_pred = mri_data_generator(val_df, batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)
    predictions = model.predict(val_gen_for_pred, steps=validation_steps)
    num_val_samples = validation_steps * batch_size
    true_ages = val_df.iloc[:num_val_samples]['age'].values

    plt.figure(figsize=(8, 6))
    plt.scatter(true_ages, predictions.flatten(), color='b', alpha=0.6, label='Predicted Age')

    min_age = min(true_ages.min(), predictions.min())
    max_age = max(true_ages.max(), predictions.max())
    plt.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=2, label='Ideal Prediction')
    plt.xlabel("True Age (years)", fontsize=14)
    plt.ylabel("Predicted Age (years)", fontsize=14)
    plt.title("Predicted vs. True Age on Validation Set", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    main()
