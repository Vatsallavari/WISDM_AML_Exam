import pandas as pd
from sklearn.utils import resample

# File paths
input_file = "WISDM_raw.csv"
cleaned_file = "WISDM_cleaned_balanced.csv"

# Step 1: Clean and keep only valid rows with 6 columns
print("[1] Cleaning rows...")
clean_rows = []
with open(input_file, "r") as infile:
    for line in infile:
        line = line.strip().rstrip(";")
        if line.count(",") == 5:
            clean_rows.append(line)

# Step 2: Create DataFrame
df = pd.DataFrame([row.split(",") for row in clean_rows],
                  columns=["User", "Activity", "Timestamp", "X", "Y", "Z"])

# Step 3: Convert types
df["User"] = pd.to_numeric(df["User"], errors="coerce")
df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
df["X"] = pd.to_numeric(df["X"], errors="coerce")
df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
df["Z"] = pd.to_numeric(df["Z"], errors="coerce")

# Step 4: Drop rows with missing values
print(f"[2] Dropping rows with nulls: {df.isnull().sum().sum()} total missing values found")
df.dropna(inplace=True)

# Step 5: Balance classes (Downsample majority classes)
print("[3] Balancing dataset by downsampling...")
min_count = df["Activity"].value_counts().min()
balanced_df = pd.concat([
    resample(df[df["Activity"] == activity],
             replace=False,
             n_samples=min_count,
             random_state=42)
    for activity in df["Activity"].unique()
])

# Optional: Shuffle the final balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 6: Save cleaned dataset
balanced_df.to_csv(cleaned_file, index=False)
print(f"[✅] Cleaned and balanced dataset saved to: {cleaned_file}")

# Step 7: Print class distribution
print("\nFinal class distribution:")
print(balanced_df["Activity"].value_counts())
