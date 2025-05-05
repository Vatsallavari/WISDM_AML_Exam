import pandas as pd

# Paths
input_file = "WISDM_cleaned_balanced.csv"
output_report = "clean_analysis.txt"

# Load dataset
df = pd.read_csv(input_file)

# Ensure data types are correct
df["User"] = pd.to_numeric(df["User"], errors="coerce")
df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
df["X"] = pd.to_numeric(df["X"], errors="coerce")
df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
df["Z"] = pd.to_numeric(df["Z"], errors="coerce")

# Generate analysis and write to file
with open(output_report, "w", encoding="utf-8") as f:
    f.write("=== WISDM Cleaned & Balanced Dataset Analysis ===\n\n")

    f.write(f"📊 Shape of dataset: {df.shape}\n\n")

    f.write("🔹 First 5 Rows:\n")
    f.write(df.head().to_string(index=False) + "\n\n")

    f.write("📈 Balanced Activity Counts:\n")
    f.write(df["Activity"].value_counts().to_string() + "\n\n")

    f.write("🧪 Missing Values per Column:\n")
    f.write(df.isnull().sum().to_string() + "\n\n")

    f.write("📐 Statistical Summary:\n")
    f.write(df.describe().to_string() + "\n")

print(f"[✅] Analysis written to {output_report}")
