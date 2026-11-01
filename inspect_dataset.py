
import pandas as pd
from sklearn.datasets import load_breast_cancer


# Load the dataset
data = load_breast_cancer(as_frame=True)

df = data.frame


# Display dataset information
print("=" * 45)
print("BREAST CANCER DATASET")
print("=" * 45)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")

class_counts = df["target"].map(
    dict(enumerate(data.target_names))
).value_counts()

print(class_counts)

print("\nFeature names:")
print(data.feature_names)

print("\nDataset inspection completed!")
