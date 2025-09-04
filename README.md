Sleep Quality Classification with Somnox & Smartwatch Data

This project explores how wearable device data collected via the Somnox sleep robot and a smartwatch can be used to classify users’ sleep quality using machine learning models. After identifying the most informative feature: 'Latency to Persistent Sleep'  we trained and compared multiple models (Random Forest, AdaBoost, XGBoost, Neural Networks) within an automated pipeline to determine the best-performing approach for sleep quality classification.

🎯 Goal

To predict Latency to Persistent Sleep based on physiological and behavioral signals, and to evaluate how well machine learning models can classify sleep quality into five levels:
Very Low · Low · Medium · High · Very High

🛠️ Workflow
	•	Data inspection & cleaning (thresholds for missing/zero values)
	•	Outlier handling with Winsorization
	•	SQ-feature filtering (to prevent leakage)
	•	Feature normalization (StandardScaler)
	•	Target binning into 5 categories
	•	Recursive Feature Elimination (RFE)
	•	Automated model training & evaluation (switch between algorithms)
	•	SHAP analysis for model interpretability

🔍 Models Used
	•	Random Forest
	•	AdaBoost
	•	XGBoost
	•	Neural Networks (baseline MLP)
	•	Automated framework allows easy switching between models to compare performance
	•	Evaluation: Accuracy, Precision, Recall, F1-score
	•	Feature importance: SHAP summary plots

⚠️ Dataset Note

Due to limited data size, especially in the test set, classification metrics are unstable:
	•	Test set size: 7 examples
	•	Accuracy: ~29%
	•	Some classes have only 1–2 samples → metrics are not representative

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
