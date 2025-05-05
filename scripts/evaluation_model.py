import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import joblib

# Paths
input_file = "dataset\WISDM_cleaned_balanced.csv"
output_model = "wisdm_rf_model.pkl"
output_report = "analysis\evaluation.txt"

# Load dataset
df = pd.read_csv(input_file)

# Features and target
X = df[["X", "Y", "Z"]]
y = df["Activity"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Save model
joblib.dump(model, output_model)

# Save evaluation results to file
with open(output_report, "w", encoding="utf-8") as f:
    f.write("=== WISDM Activity Classifier Evaluation ===\n\n")
    f.write(f"📊 Accuracy       : {acc:.4f}\n")
    f.write(f"🎯 F1 Score       : {f1:.4f}\n")
    f.write(f"🔍 Precision      : {precision:.4f}\n")
    f.write(f"📈 Recall         : {recall:.4f}\n\n")
    
    f.write("🧩 Confusion Matrix:\n")
    f.write(pd.DataFrame(conf_matrix).to_string() + "\n\n")
    
    f.write("🧾 Classification Report:\n")
    f.write(class_report)

print(f"[✅] Model trained and saved as '{output_model}'")
print(f"[📄] Evaluation report saved to '{output_report}'")
