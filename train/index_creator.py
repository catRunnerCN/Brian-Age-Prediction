"""
This file creates .csv files that contains MRI file path, participant's age, participant's id.
"""
import os
import pandas as pd

def main():
    df = pd.read_csv("F:\\Dataset\\val\\val_labels\\participants.tsv", sep="\t")
    print("Original amount of data: ", df.shape)

    df = df.dropna(subset=["age"])
    print("Amount of data after filtering data without label age:", df.shape)

    def make_vbm_path(pid):
        fname = f"sub-{pid}_preproc-quasiraw_T1w.npy"
        return os.path.join("F:\\Dataset\\val\\val_quasiraw", fname)

    df["file_path"] = df["participant_id"].apply(make_vbm_path)

    df["exists"] = df["file_path"].apply(os.path.exists)
    df = df[df["exists"] == True].drop(columns=["exists"])
    print("Amount of data after filtering non-existed data: ", df.shape)

    df = df[["participant_id", "age", "file_path"]]

    output_file = "../test/val_quasiraw_index.csv"
    df.to_csv(output_file, index=False)
    print(f"Success. File generated {output_file}, Final amount of data: {df.shape}")

if __name__ == "__main__":
    main()
