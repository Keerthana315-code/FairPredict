
import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path

# Project root directory
project_path = Path(__file__).resolve().parent.parent

# Dataset save location
dataset_path = project_path / "dataset" / "adult.csv"

print("Downloading Adult Income Dataset...")

# Fetch dataset from UCI repository
adult = fetch_ucirepo(id=2)

# Get features and target
X = adult.data.features
y = adult.data.targets

# Combine features and target
data = X.copy()
data["income"] = y.iloc[:, 0]

# Save dataset as CSV
data.to_csv(dataset_path, index=False)

print("\nDataset downloaded successfully!")
print("Saved location:", dataset_path)
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset columns:")
print(data.columns.tolist())
