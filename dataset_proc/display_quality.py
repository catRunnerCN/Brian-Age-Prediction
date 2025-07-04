import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mri_quality_stats.csv")

print(df.describe())

plt.figure(figsize=(8, 4))
plt.hist(df["mean"], bins=50, edgecolor="k")
plt.title("Histogram of MRI Mean Values")
plt.xlabel("Mean")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(df["std"], bins=50, edgecolor="k")
plt.title("Histogram of MRI Std Values")
plt.xlabel("Std")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 4))
plt.boxplot(df["std"])
plt.title("Boxplot of MRI Std Values")
plt.ylabel("Std")
plt.show()
