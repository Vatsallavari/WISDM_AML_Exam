import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


def load_features(file_path="dataset\dataset.txt"):
    """Load and preprocess raw WISDM data."""
    print(f"[📂] Loading: {file_path}")
    data = []
    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 6:
                    try:
                        user = int(parts[0])
                        activity = parts[1]
                        timestamp = float(parts[2])
                        x = float(parts[3])
                        y = float(parts[4])
                        z_str = parts[5].strip().replace(";", "")  # <-- Fix is here
                        z = float(z_str)
                        data.append([user, activity, timestamp, x, y, z])
                    except Exception as e:
                        print(f"[WARN] Skipping line {i}: {e}")
                else:
                    print(f"[WARN] Malformed line {i}: {line}")
    except Exception as e:
        print(f"[❌] Failed to open file: {e}")

    df = pd.DataFrame(data, columns=["user_id", "activity", "timestamp", "x", "y", "z"])
    print(f"[INFO] Parsed {len(df)} valid rows")
    return df



def extract_features(df, window_size=80, overlap=40):
    features, labels = [], []
    grouped = df.groupby(['user_id', 'activity'])

    for (user, activity), group in grouped:
        group = group.sort_values("timestamp")
        i = 0
        while i + window_size <= len(group):
            window = group.iloc[i:i+window_size]
            x, y, z = window['x'], window['y'], window['z']
            mag = np.sqrt(x**2 + y**2 + z**2)

            feats = [
                x.mean(), y.mean(), z.mean(),
                x.std(), y.std(), z.std(),
                x.min(), y.min(), z.min(),
                x.max(), y.max(), z.max(),
                x.corr(y), x.corr(z), y.corr(z),
                mag.mean(), mag.std()
            ]
            feats = [0 if np.isnan(val) else val for val in feats]
            features.append(feats)
            labels.append(activity)
            i += window_size - overlap

    return pd.DataFrame(features), pd.Series(labels)


def fallback_dataset():
    print("[⚠️] Falling back to synthetic dataset...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=17, n_classes=6, random_state=42)
    return pd.DataFrame(X), pd.Series([f"class_{i}" for i in y])


def evaluate_models(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='f1_weighted').mean()
        cv_acc = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy').mean()
        results.append({
            "Model": name,
            "Test Accuracy": acc,
            "CV Accuracy": cv_acc,
            "CV F1 Score": f1
        })

        print(f"\n=== {name} ===")
        print(f"Test Accuracy     : {acc:.4f}")
        print(f"CV Accuracy (avg) : {cv_acc:.4f}")
        print(f"CV F1 Score (avg) : {f1:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Optional: Confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).sort_values(by="CV Accuracy", ascending=False)


def main():
    print("[🚀] Starting Model Comparison Pipeline...")
    df = load_features("dataset.txt")

    if len(df) < 500:
        print("[❗] Not enough data — trying smaller window or fallback...")
        X, y = extract_features(df, window_size=40, overlap=20)
    else:
        X, y = extract_features(df)

    if X.empty or len(y) == 0:
        X, y = fallback_dataset()

    print(f"[📊] Final shape — Samples: {len(X)}, Features: {X.shape[1]}")
    results_df = evaluate_models(X, y)

    print("\n=== 📈 Final Model Comparison ===")
    print(results_df)
    results_df.to_csv("model_comparison_report.csv", index=False)
    print("[💾] Comparison saved to model_comparison_report.csv")


if __name__ == "__main__":
    main()
