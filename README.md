# WISDM Activity Recognition Project

## Overview
This project implements and analyzes various machine learning models for activity recognition using the WISDM (Wireless Sensor Data Mining) dataset. The dataset contains accelerometer data collected from smartphones to recognize different human activities.

---

## Dataset Information
The dataset is sourced from the WISDM Lab at Fordham University. For more details, visit: [WISDM Dataset](https://www.cis.fordham.edu/wisdm/dataset.php)

---

### Dataset Statistics
- Number of examples: 1,098,207
- Number of attributes: 6
- Activities:
  - Walking (38.6%)
  - Jogging (31.2%)
  - Upstairs (11.2%)
  - Downstairs (9.1%)
  - Sitting (5.5%)
  - Standing (4.4%)

---

## Project Structure
```
.
├── analysis/        # Analysis scripts and results
├── scripts/         # Implementation scripts
├── images/          # Visualization results
└── exam/           # Additional resources
```

---

## Implemented Models
The project implements and compares several machine learning models for activity recognition:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. Random Forest

---

## Results
The performance of each model is visualized in the `images/` directory:
- `logistic_regression.png`: Logistic Regression results
- `decision_tree.png`: Decision Tree results
- `KNN.png`: K-Nearest Neighbors results
- `SVM.png`: Support Vector Machine results
- `random_forest.png` and `random_forest_1.png`: Random Forest results
