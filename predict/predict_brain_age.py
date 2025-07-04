#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file draws a user interface for users to load model, load MRI data, view MRI slices and predict brain age.
"""

import os
import numpy as np
import tensorflow as tf
import skimage.transform
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MODEL_PATH = r"F:\Brain_Age_Estimation\brain_age_estimation\code\train\brain_age_estimation_model"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Default model is loaded: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Fail to load default model: {e}")

current_volume = None
canvas = None
plot_frame = None

def load_mri_file(file_path):
    data = np.load(file_path, mmap_mode='r')
    volume = data[0, 0, ...].astype(np.float32)
    m = volume.mean()
    s = volume.std() + 1e-8
    volume = (volume - m) / s
    return volume

def predict_brain_age(volume):
    vol_4d = np.expand_dims(volume, axis=-1)
    vol_5d = np.expand_dims(vol_4d, axis=0)
    prediction = model.predict(vol_5d)
    predicted_age = prediction[0, 0]
    return predicted_age

def display_slices_embedded(volume):
    global canvas, plot_frame
    if canvas is not None:
        canvas.get_tk_widget().destroy()

    d, h, w = volume.shape
    sagittal = volume[d // 2, :, :]
    coronal  = volume[:, h // 2, :]
    axial    = volume[:, :, w // 2]

    fig = Figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(sagittal, cmap='gray')
    ax1.set_title(f"Sagittal (x={d // 2})")
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(coronal, cmap='gray')
    ax2.set_title(f"Coronal (y={h // 2})")
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(axial, cmap='gray')
    ax3.set_title(f"Axial (z={w // 2})")
    ax3.axis('off')

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def choose_file():
    global current_volume
    file_path = filedialog.askopenfilename(title="Please choose an MRI file", filetypes=[("Numpy files", "*.npy")])
    if file_path:
        file_label.config(text=file_path)
        current_volume = load_mri_file(file_path)
        prediction_label.config(text="")

def predict_action():
    global current_volume
    if current_volume is None:
        prediction_label.config(text="Please choose an MRI file first!")
    elif model is None:
        prediction_label.config(text="Please choose a valid model folder!")
    else:
        predicted_age = predict_brain_age(current_volume)
        prediction_label.config(text=f"Predicted Brain Age: {predicted_age:.2f}")

def show_slices_action():
    global current_volume
    if current_volume is None:
        messagebox.showwarning("Warning", "Please choose an MRI file and predict!")
    else:
        display_slices_embedded(current_volume)

def choose_model_folder():
    global MODEL_PATH, model, model_label
    folder_path = filedialog.askdirectory(title="Please choose the model folder")
    if folder_path:
        try:
            model = tf.keras.models.load_model(folder_path)
            MODEL_PATH = folder_path
            model_label.config(text=f"Model folder: {folder_path}")
            messagebox.showinfo("Success", "Successfully load model!")
            print(f"Model loaded: {folder_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Fail to load model: {e}")

def create_gui():
    global file_label, prediction_label, plot_frame, model_label
    root = tk.Tk()
    root.title("Brain-age Estimation")
    root.geometry("900x650")
    root.configure(bg="#f0f0f0")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TButton", font=("Helvetica", 12), padding=6)
    style.configure("TLabel", font=("Helvetica", 12), background="#f0f0f0")
    style.configure("TFrame", background="#f0f0f0")

    title_label = ttk.Label(root, text="BAE System", font=("Helvetica", 18, "bold"))
    title_label.pack(pady=10)

    top_container = ttk.Frame(root)
    top_container.pack(padx=20, pady=10, fill=tk.X)

    left_frame = ttk.Frame(top_container)
    left_frame.grid(row=0, column=0, sticky="w", padx=10)

    file_frame = ttk.Frame(left_frame)
    file_frame.pack(pady=5, fill=tk.X)
    choose_button = ttk.Button(file_frame, text="Choose MRI file", command=choose_file)
    choose_button.pack(side=tk.LEFT, padx=5)
    file_label = ttk.Label(file_frame, text="File not chosen", wraplength=350)
    file_label.pack(side=tk.LEFT, padx=5)

    model_frame = ttk.Frame(left_frame)
    model_frame.pack(pady=5, fill=tk.X)
    model_button = ttk.Button(model_frame, text="Choose model folder", command=choose_model_folder)
    model_button.pack(side=tk.LEFT, padx=5)
    model_label = ttk.Label(model_frame, text=f"Current model folder: {MODEL_PATH}", wraplength=350)
    model_label.pack(side=tk.LEFT, padx=5)

    right_frame = ttk.Frame(top_container)
    right_frame.grid(row=0, column=1, sticky="e", padx=10)
    predict_button = ttk.Button(right_frame, text="Predict Brain Age", command=predict_action)
    predict_button.pack(pady=5, fill=tk.X)
    prediction_label = ttk.Label(right_frame, text="", font=("Helvetica", 14))
    prediction_label.pack(pady=5)

    mid_frame = ttk.Frame(root)
    mid_frame.pack(pady=10)
    slices_button = ttk.Button(mid_frame, text="Show MRI Slices", command=show_slices_action)
    slices_button.pack()

    plot_frame = ttk.Frame(root, borderwidth=2, relief="sunken")
    plot_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
