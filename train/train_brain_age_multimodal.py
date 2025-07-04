"""
decision_level_fusion_generator.py

目标级融合示例：
1. 分别训练两个独立的 3D CNN 模型（一个基于 VBM 数据，一个基于 Quasiraw 数据），
   使用生成器按批加载数据以降低内存占用（不对原始数据进行 resize）；
2. 在测试时分别预测，然后对两个模型的预测结果进行简单平均融合得到最终预测；
3. 分别输出 VBM 模型、Quasiraw 模型和融合后的测试集评估结果（MSE 和 MAE）。

索引表 "my_index_vbm_quasiraw.csv" 应包含以下列：
    participant_id, age, file_path_vbm, file_path_quasiraw
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import skimage.transform  # 本例不进行resize

###############################################################################
# 1. 单模态生成器函数
###############################################################################
def modality_generator(df, modality, batch_size=2, do_resize=False, new_shape=None, shuffle=True, infinite=True):
    """
    按批次加载指定模态的数据，并返回 (X_batch, y_batch)。

    参数：
      df: pandas DataFrame，必须包含 "file_path_{modality}" 和 "age" 列。
          例如，当 modality 为 "vbm" 时，要求有 "file_path_vbm" 列，
          当 modality 为 "quasiraw" 时，要求有 "file_path_quasiraw" 列。
      modality: 字符串，取值 "vbm" 或 "quasiraw"，指定加载哪种数据。
      batch_size: 每个批次加载的样本数量，默认为 2。
      do_resize: 是否对数据进行下采样。若 True，则将数据 resize 到 new_shape。
      new_shape: 新的目标尺寸 (D, H, W)；仅在 do_resize 为 True 时生效。
      shuffle: 是否在每个 epoch 开始时打乱 df 的顺序，默认为 True。
      infinite: 是否无限循环生成数据（适用于 model.fit(generator=...)），默认为 True；若 False，则遍历完 df 后结束。

    返回：
      yield (X_batch, y_batch)，其中：
         X_batch: NumPy 数组，形状为 (batch_size, D, H, W, 1)（D,H,W 分别为该模态的原始尺寸或经过下采样后的尺寸）。
         y_batch: NumPy 数组，形状为 (batch_size,)，为对应的年龄标签。
    """
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
            file_col = f"file_path_{modality}"
            file_path = row[file_col]
            age = row["age"]
            # 读取 .npy 文件；假设形状为 (1,1,D,H,W)
            data = np.load(file_path, mmap_mode='r')
            vol = data[0, 0, ...].astype(np.float32)  # 取出 3D 数据
            if do_resize and new_shape is not None:
                vol = skimage.transform.resize(vol, new_shape, order=1, preserve_range=True).astype(np.float32)
            # 单样本标准化 (z-score)
            m = vol.mean()
            s = vol.std() + 1e-8
            vol = (vol - m) / s
            # 扩展通道维度 => (D, H, W, 1)
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

###############################################################################
# 2. 模型构建函数（简单的3D CNN）
###############################################################################
def create_model(input_shape):
    """
    构建一个简单的 3D CNN 模型，用于回归预测脑龄。

    参数：
      input_shape: 一个元组，表示输入数据的形状，如 (D, H, W, 1)。
                 对于 VBM 模态，例如 (121,145,121,1)；
                 对于 Quasiraw 模态，例如 (182,218,182,1)。

    返回：
      一个编译好的 Keras 模型对象，使用 Adam 优化器、均方误差损失，并监控 MAE。
    """
    model = models.Sequential()
    model.add(layers.Conv3D(16, (3,3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2,2,2)))
    model.add(layers.Conv3D(32, (3,3,3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2,2,2)))
    model.add(layers.Conv3D(64, (3,3,3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2,2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

###############################################################################
# 3. 主程序：目标级融合 + 使用生成器
###############################################################################
def main():
    # --- A. 读取多模态索引表 ---
    # 索引表 "my_index_vbm_quasiraw.csv" 应包含：participant_id, age, file_path_vbm, file_path_quasiraw
    index_file = "my_index_vbm_quasiraw.csv"
    df_all = pd.read_csv(index_file)
    print("Total samples in index:", len(df_all))

    # --- B. 拆分训练集和测试集 ---
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
    print("Train samples:", len(train_df), "Test samples:", len(test_df))

    # --- C. 构建生成器 ---
    batch_size = 2
    # 构建 VBM 和 Quasiraw 的训练生成器（不进行 resize，保留原始分辨率）
    gen_vbm_train = modality_generator(train_df, modality="vbm", batch_size=batch_size, do_resize=False, shuffle=True, infinite=True)
    gen_qr_train = modality_generator(train_df, modality="quasiraw", batch_size=batch_size, do_resize=False, shuffle=True, infinite=True)

    # 构建测试生成器
    gen_vbm_test = modality_generator(test_df, modality="vbm", batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)
    gen_qr_test = modality_generator(test_df, modality="quasiraw", batch_size=batch_size, do_resize=False, shuffle=False, infinite=False)

    # --- D. 获取生成器输出以确定各模态输入尺寸 ---
    sample_vbm, _ = next(gen_vbm_train)
    sample_qr, _ = next(gen_qr_train)
    input_shape_vbm = sample_vbm.shape[1:]  # 例如 (121,145,121,1)
    input_shape_qr = sample_qr.shape[1:]      # 例如 (182,218,182,1)
    print("VBM input shape:", input_shape_vbm)
    print("Quasiraw input shape:", input_shape_qr)

    # --- E. 构建各单模态模型 ---
    model_vbm = create_model(input_shape_vbm)
    model_qr = create_model(input_shape_qr)
    print("VBM Model Summary:")
    model_vbm.summary()
    print("Quasiraw Model Summary:")
    model_qr.summary()

    # --- F. 训练各单模态模型 ---
    epochs = 20
    steps_per_epoch = len(train_df) // batch_size
    model_vbm.fit(gen_vbm_train, steps_per_epoch=steps_per_epoch, epochs=epochs)
    model_qr.fit(gen_qr_train, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # --- G. 在测试集上分别进行预测 ---
    test_steps = len(test_df) // batch_size
    pred_vbm = model_vbm.predict(gen_vbm_test, steps=test_steps)
    pred_qr = model_qr.predict(gen_qr_test, steps=test_steps)

    # --- H. 目标级融合：简单平均两个模型的预测结果 ---
    final_pred = (pred_vbm + pred_qr) / 2.0

    # --- I. 评估结果 ---
    # 从 test_df 中直接获取真实标签
    y_test = test_df["age"].values.astype(np.float32)
    mse_vbm = np.mean((pred_vbm.flatten() - y_test)**2)
    mae_vbm = np.mean(np.abs(pred_vbm.flatten() - y_test))
    mse_qr = np.mean((pred_qr.flatten() - y_test)**2)
    mae_qr = np.mean(np.abs(pred_qr.flatten() - y_test))
    mse_fusion = np.mean((final_pred.flatten() - y_test)**2)
    mae_fusion = np.mean(np.abs(final_pred.flatten() - y_test))

    print(f"VBM Model: Test MSE: {mse_vbm:.4f}, Test MAE: {mae_vbm:.4f}")
    print(f"Quasiraw Model: Test MSE: {mse_qr:.4f}, Test MAE: {mae_qr:.4f}")
    print(f"Fusion Result: Test MSE: {mse_fusion:.4f}, Test MAE: {mae_fusion:.4f}")

if __name__ == "__main__":
    main()
