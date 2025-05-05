import pandas as pd

# File paths
input_file = "WISDM_raw.csv"
cleaned_file = "WISDM_cleaned.csv"
output_report = "analysis.txt"

# Step 1: Clean raw data
with open(input_file, "r") as infile, open(cleaned_file, "w", encoding="utf-8") as outfile:
    total, kept = 0, 0
    for line in infile:
        total += 1
        line = line.strip().rstrip(";")
        if line.count(",") == 5:  # Valid line with 6 columns
            outfile.write(line + "\n")
            kept += 1

print(f"[✔] Cleaned {kept} of {total} rows. Saved to {cleaned_file}")

# Step 2: Load and assign column names
df = pd.read_csv(cleaned_file, header=None)
df.columns = ["User", "Activity", "Timestamp", "X", "Y", "Z"]

# Step 3: Clean and convert Z column
df["Z"] = pd.to_numeric(df["Z"], errors="coerce")

# Step 4: Write analysis to file (with UTF-8 encoding to support emojis)
with open(output_report, "w", encoding="utf-8") as f:
    f.write("=== WISDM Dataset Analysis ===\n\n")
    f.write(f"📊 Shape of dataset: {df.shape}\n\n")

    f.write("🔹 First 5 Rows:\n")
    f.write(df.head().to_string(index=False) + "\n\n")

    f.write("📈 Activity Counts:\n")
    f.write(df["Activity"].value_counts().to_string() + "\n\n")

    f.write("🧪 Missing Values per Column:\n")
    f.write(df.isnull().sum().to_string() + "\n\n")

    f.write("📐 Statistical Summary:\n")
    f.write(df.describe().to_string() + "\n")

print(f"[✅] Analysis written to {output_report}")
