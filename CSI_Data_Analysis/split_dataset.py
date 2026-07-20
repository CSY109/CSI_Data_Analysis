import numpy as np
from sklearn.model_selection import train_test_split
import os

data_dir = r"f:\CSI_Data_Analysis\DataNpy20260720"

# Load data
X = np.load(os.path.join(data_dir, "X_custom.npy"))
y = np.load(os.path.join(data_dir, "y_custom.npy"))

print(f"Original dataset shape: X={X.shape}, y={y.shape}")

# Ensure there's enough data to split
if len(X) < 10:
    print("Dataset too small for meaningful split.")
    exit(1)

# Split into train and temp (70% train, 30% temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split temp into val and test (50% val, 50% test -> 15% each of total)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Backup original
os.rename(os.path.join(data_dir, "X_custom.npy"), os.path.join(data_dir, "X_all.npy"))
os.rename(os.path.join(data_dir, "y_custom.npy"), os.path.join(data_dir, "y_all.npy"))

# Save splits
np.save(os.path.join(data_dir, "X_custom.npy"), X_train)
np.save(os.path.join(data_dir, "y_custom.npy"), y_train)

np.save(os.path.join(data_dir, "X_val.npy"), X_val)
np.save(os.path.join(data_dir, "y_val.npy"), y_val)

np.save(os.path.join(data_dir, "X_test.npy"), X_test)
np.save(os.path.join(data_dir, "y_test.npy"), y_test)

print("Split completed successfully!")
print(f"Total samples: {len(X)}")
print(f"Train samples (X_custom.npy): {len(X_train)}")
print(f"Val samples (X_val.npy): {len(X_val)}")
print(f"Test samples (X_test.npy): {len(X_test)}")
