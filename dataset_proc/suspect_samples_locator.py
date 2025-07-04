import pandas as pd

df = pd.read_csv("mri_quality_stats.csv")

std_low = 0.15
std_high = 0.30

df_suspect = df[(df["std"] < std_low) | (df["std"] > std_high)]

print("Amount of suspect samples:", len(df_suspect))
print(df_suspect)

df_suspect.to_csv("suspect_samples.csv", index=False)
