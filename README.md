# Sleep Quality Classification with Somnox & Smartwatch Data

This project explores how wearable device data collected via the Somnox sleep robot and the Smartwatch can be used to classify users' sleep quality using machine learning models.

## Goal

To predict **Latency to Persistent Sleep** using physiological and behavioral signals and evaluate how well machine learning models can classify sleep quality levels (e.g., Very Low to Very High).

##  Workflow

- Data inspection & cleaning (with thresholds for missing/zero values)
- Winsorization-based outlier handling
- SQ-feature filtering (to prevent leakage)
- Feature normalization (StandardScaler)
- Target binning into 5 classes: *Very Low, Low, Medium, High, Very High*
- Recursive Feature Elimination (RFE)
- Model training with Random Forest, XGBoost, etc.
- Evaluation on validation & test sets
- SHAP analysis for model interpretability

## 🔍 Models Used

- Random Forest (default + hyperparameter tuned)
- XGBoost (planned)
- Classifier evaluation: Accuracy, Precision, Recall, F1-score
- Feature importance: SHAP summary plots

## ⚠️ Note on Dataset Size

! Due to limited data availability, especially in the test set, classification metrics are currently unstable:

```text
Test Set Size: 7 examples  
Accuracy: ~29%  
Some classes have 1-2 samples → metrics are unreliable
