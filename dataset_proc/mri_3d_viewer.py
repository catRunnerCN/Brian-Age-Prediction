"""
mri_3d_viewer.py
This file downsample the selected MRI and display the 3D model of this MRI in browser.
"""

import numpy as np
import plotly.graph_objects as go
import skimage.transform

def load_volume(file_path):
    data = np.load(file_path, mmap_mode='r')
    volume = data[0, 0, ...].astype(np.float32)
    return volume

def downsample(volume, new_shape=(64,64,64)):
    vol_ds = skimage.transform.resize(
        volume,
        new_shape,
        order=1,
        preserve_range=True
    ).astype(np.float32)
    return vol_ds

def plot_volume(volume, opacity=0.1, surface_count=21):
    d, h, w = volume.shape

    print(f"Volume shape after downsample: {volume.shape}")
    vol_min, vol_max = volume.min(), volume.max()
    print(f"Volume min={vol_min:.4f}, max={vol_max:.4f}")

    if np.isclose(vol_min, vol_max):
        print("Warning: The voxel values in the data are almost the same, and visible structures may not be rendered.")

    x, y, z = np.mgrid[0:d, 0:h, 0:w]

    isomin = vol_min
    isomax = vol_max

    fig = go.Figure(data=go.Volume(
        x = x.flatten(),
        y = y.flatten(),
        z = z.flatten(),
        value = volume.flatten(),
        isomin = isomin,
        isomax = isomax,
        opacity = opacity,
        surface_count = surface_count,
        colorscale = 'gray'
    ))

    fig.update_layout(scene=dict(aspectmode='data'),
                      title="3D MRI Volume Rendering (Downsampled)")
    fig.show(renderer='browser')

def main():
    file_path = r"F:\Dataset\train\train_quasiraw\sub-100053248969_preproc-quasiraw_T1w.npy"

    volume = load_volume(file_path)
    print("Original volume shape:", volume.shape)

    ds_shape = (64, 64, 64)
    volume_ds = downsample(volume, new_shape=ds_shape)

    plot_volume(volume_ds, opacity=0.1, surface_count=16)

if __name__ == "__main__":
    main()
