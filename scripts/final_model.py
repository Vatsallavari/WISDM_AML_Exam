import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings('ignore')


# Function to load the dataset from local file
def load_dataset(file_path='dataset\dataset.txt'):
    print(f"[📂] Loading dataset from {file_path}...")
    try:
        with open(file_path, 'r') as file:
            data = file.read()
        return data
    except Exception as e:
        print(f"[❌] Failed to load file: {e}")
        return None


# Function to preprocess raw WISDM data
def preprocess_data(data):
    print("[🔧] Preprocessing data...")
    processed_data = []
    for line in data.strip().split('\n'):
        try:
            if line.endswith(';'):
                line = line[:-1]
            parts = line.strip().split(',')
            if len(parts) >= 6:
                processed_data.append([
                    int(parts[0]),
                    parts[1],
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5])
                ])
        except:
            continue
    columns = ['user_id', 'activity', 'timestamp', 'x_accel', 'y_accel', 'z_accel']
    return pd.DataFrame(processed_data, columns=columns)


# Feature extraction from accelerometer data
def extract_features(df, window_size=80, overlap=40):
    print("[📊] Extracting features...")
    features = []
    labels = []
    grouped = df.groupby(['user_id', 'activity'])

    for (user, activity), group in grouped:
        group = group.sort_values('timestamp')
        i = 0
        while i + window_size <= len(group):
            window = group.iloc[i:i+window_size]
            x = window['x_accel']
            y = window['y_accel']
            z = window['z_accel']
            mag = np.sqrt(x**2 + y**2 + z**2)

            feature_vector = [
                x.mean(), y.mean(), z.mean(),
                x.std(), y.std(), z.std(),
                x.max(), y.max(), z.max(),
                x.min(), y.min(), z.min(),
                x.corr(y), x.corr(z), y.corr(z),
                mag.mean(), mag.std()
            ]
            feature_vector = [0 if np.isnan(val) else val for val in feature_vector]

            features.append(feature_vector)
            labels.append(activity)
            i += (window_size - overlap)

    feature_columns = [
        'x_mean', 'y_mean', 'z_mean',
        'x_std', 'y_std', 'z_std',
        'x_max', 'y_max', 'z_max',
        'x_min', 'y_min', 'z_min',
        'xy_corr', 'xz_corr', 'yz_corr',
        'mag_mean', 'mag_std'
    ]

    return pd.DataFrame(features, columns=feature_columns), pd.Series(labels)


# Train and evaluate models
def train_and_evaluate(X, y):
    print("[🧠] Training and evaluating models...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=le.classes_)

    print(f"\n🎯 Random Forest Accuracy: {acc:.4f}")
    print("📈 Classification Report:\n", cr)

    results['random_forest'] = {
        'model': clf,
        'accuracy': acc,
        'conf_matrix': cm,
        'report': cr
    }

    # Cross-Validation
    print("\n🔁 Performing cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv)
    print(f"✅ CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return results


# Main driver
def main(file_path='dataset\dataset.txt'):
    raw_data = load_dataset(file_path)

    if not raw_data:
        print("[❌] No data loaded.")
        return

    df = preprocess_data(raw_data)
    print(f"[📄] Samples: {len(df)}, Users: {df['user_id'].nunique()}, Activities: {df['activity'].nunique()}")
    print(df.head())

    X, y = extract_features(df)
    print(f"[📐] Feature shape: {X.shape}")

    results = train_and_evaluate(X, y)

    best_acc = results['random_forest']['accuracy']
    print(f"\n🏁 Best model: Random Forest with accuracy {best_acc:.4f}")

    return results


# Execute
if __name__ == "__main__":
    try:
        main("dataset\dataset.txt")
        print("[✅] Script executed successfully.")
    except Exception as e:
        print(f"[❌] Error during execution: {e}")
