import os
import pandas as pd
import shutil


def move_suspect_samples(csv_file, target_folder):
    df = pd.read_csv(csv_file)
    print(f"{len(df)} bad samples in {csv_file}")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Create target folder: {target_folder}")

    for idx, row in df.iterrows():
        file_path = row["file_path"]
        if os.path.exists(file_path):
            try:
                shutil.move(file_path, target_folder)
                print(f"Moved: {file_path}")
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")


def main():
    csv_file = r"F:\Brain_Age_Estimation\brain_age_estimation\code\dataset_proc\suspect_samples.csv"
    target_folder = r"F:\Dataset\train\suspect_samples_vbm"
    move_suspect_samples(csv_file, target_folder)


if __name__ == "__main__":
    main()
