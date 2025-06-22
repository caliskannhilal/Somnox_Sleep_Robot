#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 01:01:34 2025

@author: hilalcaliskan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor  # Added for XGBoost
from sklearn.svm import SVR  # Added for SVM
from sklearn.neural_network import MLPRegressor  # Added for Neural Networks
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import settings  # importing the settings.py file
import seaborn as sns
import shap
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor  # Added AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.inspection import permutation_importance

# Step 1: Data Inspection
def inspect_data(df_raw):
    print("Head of the DataFrame:")
    print(df_raw.head())
    print("\nInfo of the DataFrame:")
    df_raw.info()
    print("\nMissing values per column:")
    print(df_raw.isnull().sum())

# Step 2: Data Cleaning
def clean_data(df, threshold_missing, threshold_zeros):
    s_datasubject_mapping = None
    if 's_datasubject' in df.columns:
        le = LabelEncoder()
        df['s_datasubject'] = le.fit_transform(df['s_datasubject'])
        s_datasubject_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    df = df.select_dtypes(include=[np.number])
    df_cleaned = df.dropna(thresh=int((1 - threshold_missing) * len(df)), axis=1)
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).mean() >= (1 - threshold_zeros)]
    return df_cleaned.fillna(df_cleaned.mean()), s_datasubject_mapping

# Step 3: Handle Outliers

def handle_outliers(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])  # Capping at 5% extreme values
    return df

def normalize_data(df, target_column):
    """
    Normalize dataset features (excluding `s_datasubject` and `target_column`).
    """
    scaler = StandardScaler()

    # Exclude the target variable and `s_datasubject` from scaling
    features = df.drop(columns=['s_datasubject', target_column], errors="ignore").copy()

    # Handle Inf and NaNs BEFORE scaling
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(features.mean(), inplace=True)

    # Apply StandardScaler on features
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    if target_column in df.columns:
        scaled_df = pd.concat([scaled_features, df[[target_column]]], axis=1)  # Keep target unchanged
    else:
        scaled_df = scaled_features

    return scaled_df

def transform_target_to_categorical(df, target_column, bins, labels):
    """
    Convert a continuous target variable into categorical integer labels using binning.
    """
    # Ensure df is copied before modification
    df = df.copy()

    # Verify target column exists before proceeding
    if target_column not in df.columns:
        raise KeyError(f"Target column `{target_column}` not found in DataFrame!")

    # Convert target column to numeric safely
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

    # Apply binning and assign integer labels
    df[target_column] = pd.cut(df[target_column], bins=bins, labels=range(len(labels)), include_lowest=True)

    # Convert to integer type (ensuring no float or categorical issues)
    df[target_column] = df[target_column].astype(int)

    # Debugging check
    print(f" Successfully binned `{target_column}` into {len(labels)} categories.")
    print(f" Unique values in `{target_column}` after binning:", df[target_column].unique())

    return df

# Step 6: Add Noise to Target
def add_noisy_target(df, target_column, noise_factor=0.05):
    df['noisy_target'] = df[target_column] + np.random.normal(0, noise_factor * df[target_column].std(), len(df))
    return df

# Step 7: Remove SQ-related Features

def remove_sq_features(df, target_column):
    if target_column not in df.columns:
        raise KeyError(f" Target column `{target_column}` not found in DataFrame!")
    non_sq_columns = [col for col in df.columns if 'SQ' not in col or col == target_column]
    return df[non_sq_columns]

def split_data_random(df, target_column=None, y_array=None, noisy_target_column=None, 
                      test_size=0.1, val_size=0.2, random_state=42, classification=False):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): The input dataset.
    - target_column (str): Name of the target variable.
    - y_array (np.array or pd.Series, optional): If provided, use this as target instead of extracting from df.
    - noisy_target_column (str, optional): If provided, use this as target instead of target_column.
    - test_size (float): Fraction of data to use for the test set (default: 0.1).
    - val_size (float): Fraction of data to use for the validation set (default: 0.2).
    - random_state (int): Random seed for reproducibility.
    - classification (bool): Whether the task is classification.

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test (pd.DataFrame, pd.Series)
    """

    df = df.copy()  # Avoid modifying the original DataFrame

    # Select the correct target variable
    target = noisy_target_column if noisy_target_column else target_column
    if y_array is not None:
        y = y_array
        X = df  # Keep all features
    else:
        if target not in df.columns:
            raise KeyError(f"❌ Target column `{target}` is NOT found in the DataFrame!")

        # Extract target `y` and ensure it's not empty
        y = df[target].values
        if len(y) == 0:
            raise ValueError(f"❌ Target column `{target}` has NO VALUES before splitting!")

        # Drop the target column from X
        X = df.drop(columns=[target], errors="ignore")

    # Debugging: Print shapes before splitting
    print(f"✅ X shape before split: {X.shape}, y shape before split: {y.shape}")

    # Ensure dataset is large enough for splitting
    total_rows = len(X)
    if total_rows < 10:
        raise ValueError(f"❌ Not enough samples ({total_rows}) for splitting! Reduce test/val sizes.")

    # Use stratified split for classification if the target has more than one unique class
    stratify_param = y if classification and np.unique(y).size > 1 else None

    # Step 1: First split - Train+Validation vs. Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # Adjust validation split size based on remaining data
    val_size_adjusted = val_size / (1 - test_size)

    # Step 2: Second split - Train vs. Validation
    stratify_param_train = y_train_val if classification and np.unique(y_train_val).size > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=stratify_param_train
    )

    # Debugging: Print class distributions if classification
    if classification:
        print(f"\n📌 Unique classes in y_train: {np.unique(y_train)}")
        print(f"📌 Unique classes in y_val: {np.unique(y_val)}")
        print(f"📌 Unique classes in y_test: {np.unique(y_test)}")
        assert len(np.unique(y_train)) > 0, "🚨 y_train is empty after splitting!"
        assert len(np.unique(y_val)) > 0, "🚨 y_val is empty after splitting!"
        assert len(np.unique(y_test)) > 0, "🚨 y_test is empty after splitting!"

    # Final Debugging: Print dataset shapes
    print("\n✅ Final Split Results:")
    print(f"📊 X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"📊 X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")
    print(f"📊 X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 9: Recursive Feature Elimination

def perform_rfe(X_train, y_train, n_features_to_select=None, classification=True):
    """
    Perform Recursive Feature Elimination (RFE) with optimized preprocessing.
    
    - Removes `s_datasubject` if present.
    - Filters near-zero variance features.
    - Uses an efficient model for RFE (`LogisticRegression` for classification, `RandomForestRegressor` for regression).
    - Selects up to `n_features_to_select` features (default: min(10, available features)).
    - Saves and prints the top 20 selected features.

    Parameters:
    - X_train (pd.DataFrame): Training feature dataset.
    - y_train (pd.Series): Target variable (numeric or categorical).
    - n_features_to_select (int): Number of features to select (default: 10 or all available).
    - classification (bool): Whether the task is classification.

    Returns:
    - List of selected feature names.
    """

    #  Ensure `s_datasubject` is NOT in X_train
    if "s_datasubject" in X_train.columns:
        print(" Removing `s_datasubject` before RFE!")
        X_train = X_train.drop(columns=["s_datasubject"])

    #  Remove near-zero variance features
    var_thresh = VarianceThreshold(threshold=0.001)
    X_train_reduced = pd.DataFrame(var_thresh.fit_transform(X_train), 
                                   columns=X_train.columns[var_thresh.get_support()])
    print(f" Reduced features from {X_train.shape[1]} to {X_train_reduced.shape[1]} before RFE.")

    #  Define the model for RFE
    model = LogisticRegression(max_iter=500) if classification else RandomForestRegressor(n_estimators=50)

    #  Set `n_features_to_select` (defaults to at most 10 or all available)
    n_features_to_select = n_features_to_select or min(10, X_train_reduced.shape[1])

    # Apply RFE (removing 10% of features at a time)
    selector = RFE(model, n_features_to_select=n_features_to_select, step=0.1)
    selector.fit(X_train_reduced, y_train)

    #  Get selected features
    selected_features = X_train_reduced.columns[selector.support_].tolist()
    feature_ranks = dict(zip(X_train_reduced.columns, selector.ranking_))

    #  Save Feature Rankings
    feature_ranking_df = pd.DataFrame(
        sorted(feature_ranks.items(), key=lambda x: x[1]),
        columns=["Feature", "Rank"]
    )

    file_path = "feature_ranking.csv"
    feature_ranking_df.to_csv(file_path, index=False)

    #  Print Top 20 Features
    print(f"\n **Top 20 Features Selected by RFE (Saved to {file_path})**:")
    print(feature_ranking_df.head(20))

    print(f" Selected {len(selected_features)} features from {X_train.shape[1]}.")
    return selected_features

def get_model(model_type="random_forest", random_state=42, classification=False):
    """
    Returns the specified machine learning model based on the given type.
    """
    if classification:
        if model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == "xgboost":
            return XGBClassifier(n_estimators=100, random_state=random_state)
        else:
            raise ValueError(f"Unsupported classification model type: {model_type}")
    else:
        if model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_type == "xgboost":
            return XGBRegressor(n_estimators=100, random_state=random_state)
        else:
            raise ValueError(f"Unsupported regression model type: {model_type}")
            
def train_and_evaluate_random_forest(X_train, X_val, y_train, y_val, model_type="random_forest"):
    """
    Trains a Random Forest model and evaluates it on the validation set.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_val (pd.DataFrame): Validation features.
    - y_train (pd.Series): Training targets.
    - y_val (pd.Series): Validation targets.
    - model_type (str): Name for display (default: "random_forest").

    Returns:
    - model: Trained Random Forest model.
    """
    # Initialize the Random Forest model
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)

    # Validate the model
    print(f"\nEvaluating {model_type} on the validation set...")
    y_val_pred = model.predict(X_val)

    # Calculate metrics
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)

    # Display metrics
    print(f"{model_type} Validation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Visualize actual vs. predicted values
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, y_val_pred, alpha=0.7, edgecolors='k')
    plt.plot(y_val, y_val, color="red", linestyle="--", label="Ideal Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values - {model_type}")
    plt.legend()
    plt.show()

    return model

def train_and_evaluate_classifier(model, X_train, X_val, y_train, y_val, model_type="classifier"):
    """
    Trains a classifier model and evaluates it on the validation set.

    Parameters:
    - model: The classifier model to train.
    - X_train (pd.DataFrame): Training features.
    - X_val (pd.DataFrame): Validation features.
    - y_train (pd.Series): Training labels.
    - y_val (pd.Series): Validation labels.
    - model_type (str): Name for display (default: "classifier").

    Returns:
    - model: Trained classifier model.
    """
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)

    print(f"\nEvaluating {model_type} on the validation set...")
    y_val_pred = model.predict(X_val)

    # Metrics for classification
    accuracy = np.mean(y_val_pred == y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    return model

def hyperparameter_tuning(X_train, y_train, model_type="random_forest", cv=3, random_state=42):
    """
    Perform hyperparameter tuning for the specified model type using GridSearchCV.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training targets.
    - model_type (str): Type of model to use ("random_forest", "xgboost", etc.).
    - cv (int): Number of cross-validation folds (default: 3).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - best_model: The best model with optimized parameters.
    - best_params: Dictionary of the best hyperparameters.
    """
    if model_type == "random_forest":
        # model = RandomForestRegressor(random_state=random_state)
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == "xgboost":
        model = XGBRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.01, 0.1]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_with_confusion_matrix(y_true, y_pred, model_type="model", category_order=None):
    """
    Generate and visualize a confusion matrix for classification predictions.

    Parameters:
    - y_true (pd.Series or np.array): Actual class labels.
    - y_pred (pd.Series or np.array): Predicted class labels.
    - model_type (str): Model name for labeling in the output.
    - category_order (list): List of category labels in correct order (if classification).

    Returns:
    - None: Displays the confusion matrix.
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Ensure labels match category codes correctly
    if category_order:
        category_labels = {i: label for i, label in enumerate(category_order)}
        xticklabels = [category_labels[i] for i in range(len(category_order))]
        yticklabels = [category_labels[i] for i in range(len(category_order))]
    else:
        xticklabels = yticklabels = sorted(set(y_true))  # Use numerical labels if no category order

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title(f"Confusion Matrix - {model_type}")
    plt.show()
    
    
def print_class_report(y_test, y_test_pred):
    # Define category labels
    category_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    category_labels = {i: label for i, label in enumerate(category_order)}
    
    # Get only classes that appear in `y_test` and `y_test_pred`
    unique_classes = np.unique(np.concatenate([y_test, y_test_pred]))
    filtered_category_order = [category_labels[i] for i in unique_classes]
    
    print(f"Final category labels used: {filtered_category_order}")  # Debugging
    
    # Generate classification report with the correct target names
    print("\n Classification Report:\n")
    print(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=filtered_category_order))

def evaluate_with_classification_metrics(y_true, y_pred, model_type="classifier"):
    """
    Evaluate classification predictions with detailed metrics.

    Parameters:
    - y_true (pd.Series or np.array): Actual class labels.
    - y_pred (pd.Series or np.array): Predicted class labels.
    - model_type (str): Model name for labeling in the output.

    Returns:
    - None: Prints the classification metrics.
    """
    print(f"Classification Report ({model_type}):")
    print_class_report(y_true, y_pred)
    
def visualize_shap_values(model, X, feature_names, plot_type="summary", model_type="Model"):
    """
    Visualize SHAP values for feature importance.

    Parameters:
    - model: Trained model (e.g., RandomForestRegressor, XGBoost, etc.).
    - X (pd.DataFrame or np.array): Feature dataset.
    - feature_names (list): List of feature names.
    - plot_type (str): Type of SHAP plot ("summary", "bar").
    - model_type (str): Model name for labeling in the output.

    Returns:
    - None: Displays SHAP plots.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    if plot_type == "summary":
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot - {model_type}")
        plt.show()
    elif plot_type == "bar":
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Bar Plot - {model_type}")
        plt.show()
    else:
        raise ValueError("Unsupported plot_type. Use 'summary' or 'bar'.")
        
def evaluate_test_set(model, X_test, y_test, model_type="model"):
    """
    Evaluate the model on the test set.

    Parameters:
    - model: Trained model to evaluate.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target values.
    - model_type (str): Model name for labeling in the output.

    Returns:
    - None: Prints evaluation metrics and displays confusion matrix.
    """
    y_test_pred = model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\nFinal Test Results ({model_type}):")
    print(f"Mean Squared Error (MSE): {test_mse}")
    print(f"R-squared (R2): {test_r2}")
    print(f"Mean Absolute Error (MAE): {test_mae}")

    evaluate_with_confusion_matrix(y_test, y_test_pred, model_type=model_type, category_order=['Very Low', 'Low', 'Medium', 'High', 'Very High']) 
    
def plot_permutation_importance(model, X_test, y_test, model_type="model", n_repeats=10):
    """
    Computes and plots permutation importance of features.

    Parameters:
    - model: Trained model to evaluate.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target values.
    - model_type (str): Model name for labeling the plot.
    - n_repeats (int): Number of times to shuffle each feature (default: 10).

    Returns:
    - None: Displays a box plot of permutation importance.
    """
    print("\n🔍 Computing Permutation Importance...")
    
    # Compute permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)

    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        perm_importance.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx]
    )
    plt.xlabel("Decrease in Model Performance")
    plt.title(f"Permutation Feature Importance - {model_type}")
    plt.grid(True)
    plt.show()

#mainfunction        
def main(model_type="random_forest", use_total_sleep=False, add_noise=False, tune_hyperparameters=True, classification=True):
    """
    Main function to preprocess data, bin (if classification), select features, train, and evaluate the specified model.

    Parameters:
    - model_type: str, specify the type of ML model to use ("random_forest", "ada_boost", "xgboost", etc.).
    - use_total_sleep: bool, if True, use Total Sleep as the target. Otherwise, use Latency to Persistent Sleep.
    - add_noise: bool, if True, add noise to the target variable.
    - tune_hyperparameters: bool, whether to perform hyperparameter tuning (default: False).
    - classification: bool, whether to transform the target to categorical and use classification models.
    """
    # Step 1: Data Preprocessing (Load and inspect the data)
    df_raw = pd.read_csv(settings.DATA_PATH)
    inspect_data(df_raw)

    # Step 2: Handle missing values & clean dataset
    df_cleaned, s_datasubject_mapping = clean_data(
        df_raw, threshold_missing=settings.THRESHOLD_MISSING, threshold_zeros=settings.THRESHOLD_ZEROS
    )

    # Step 3: Handle outliers
    df_no_outliers = handle_outliers(df_cleaned)  # Now, outliers are capped, not removed
    print(f"Dataset size after handling outliers: {df_no_outliers.shape}")  # Debugging check  
    
    # Step 4: Define the target column dynamically
    if classification:
        target_column = "f_SQ_Fitbit_details_LatencyToPersistentSleep"  # Classification should always use Latency!
    else:
        target_column = "f_SQ_Fitbit_details_TotalSleepTime" if use_total_sleep else "f_SQ_Fitbit_details_LatencyToPersistentSleep"
        
    print(df_no_outliers.isnull().sum().sum())
    
    # Step 5: Remove SQ-Related Features (BEFORE normalization, so the target remains)
    df_non_sq = remove_sq_features(df_no_outliers, target_column)
    
    # Ensure target column is not removed
    assert target_column in df_non_sq.columns, f" Target column {target_column} was removed after removing SQ features!"

    print("Total NaNs in df_non_sq after removing SQ features:", df_non_sq.isnull().sum().sum())
    
    zero_range_cols = df_non_sq.columns[(df_non_sq.max() - df_non_sq.min()) == 0]
    print("Columns with zero range (causing NaNs in MinMaxScaler):", zero_range_cols.tolist())
    print("Hidden NaNs in df_non_sq:", df_non_sq.isnull().sum().sum())
    
    print("Checking for Inf values in df_non_sq:")
    print((df_non_sq == np.inf).sum().sum(), "positive inf values")
    print((df_non_sq == -np.inf).sum().sum(), "negative inf values")
    
    print((df_non_sq < 0).sum().sum(), "negative values found in df_non_sq")
    
   # Step 6: Apply Binning BEFORE Normalization (if classification)
    # Step 1: Check Unique Values in Target Column
    print("Unique values in target column before binning:")
    print(np.sort(df_non_sq[target_column].unique()))
    
    # Step 2: Dynamically Define Bins
    num_bins = 5  # Adjust as needed
    min_val = df_non_sq[target_column].min()
    max_val = df_non_sq[target_column].max()
    
    if min_val == max_val:
        raise ValueError(f" Target column `{target_column}` has only one unique value: {min_val}. Binning is not possible!")
    
    # Create bin edges dynamically based on min/max
    bins = np.linspace(min_val, max_val, num_bins + 1)  
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    
    print(f" Using bins: {bins}")
    
    # Step 3: Apply Binning to Target Column
    df_non_sq[target_column] = pd.cut(df_non_sq[target_column], bins=bins, labels=labels, include_lowest=True)
    
    # Step 4: Convert to Categorical
    df_non_sq[target_column] = pd.Categorical(df_non_sq[target_column], categories=labels, ordered=True)
    df_non_sq[target_column] = df_non_sq[target_column].cat.codes  # Convert categories to integer labels
    
    
    # Step 7: Normalize only the features, EXCLUDE the target variable
    df_features = df_non_sq.drop(columns=[target_column], errors="ignore")  # Drop target
    df_normalized = normalize_data(df_features, target_column)  # Normalize only features
    
    # Ensure Target Stays Integer (Enforce int64)
    df_normalized[target_column] = df_non_sq[target_column].astype(np.int64)
    
    # Debugging
    print("Final unique target values:", df_normalized[target_column].unique())
    print("Final target column dtype:", df_normalized[target_column].dtype)
    
    # Ensure target remains integer-labeled
    assert np.issubdtype(df_normalized[target_column].dtype, np.integer), "🚨 Target column was altered after normalization!"    
    
    # Debugging: Confirm target column exists before splitting
    if target_column not in df_normalized.columns:
        raise KeyError(f" Target column `{target_column}` is missing before splitting!")

    assert 's_datasubject' not in df_normalized.columns, "🚨 s_datasubject reappeared!"
    
    # Ensure target is categorical & integer-labeled BEFORE splitting
    print("📌 Unique values in target column BEFORE split:", df_non_sq[target_column].unique())
    
    # Convert to categorical and integer-encoded labels
    df_non_sq[target_column] = pd.Categorical(df_non_sq[target_column])
    df_non_sq[target_column] = df_non_sq[target_column].cat.codes  # Ensure integer labels
    
    # Debugging: Print unique values again
    print("📌 Unique values in target column AFTER encoding:", df_non_sq[target_column].unique())
    
    #  Now Split X and y AFTER binning & encoding
    X = df_normalized.drop(columns=["s_datasubject", target_column], errors="ignore")
    y = df_normalized[target_column]  #  Now it's numeric
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_random(X, y_array=y) 
        
    # Step 10: Check & fill NaNs before Feature Selection
    print("First few values of y_train:", y_train[:5])
    print("Unique classes in y_train:", np.unique(y_train))
    print("Unique classes in y_val:", np.unique(y_val))
    print("Unique classes in y_test:", np.unique(y_test))
    
    
    # Step 11: **Feature Selection (RFE)**
    selected_features = perform_rfe(
        X_train, y_train, 
        n_features_to_select=min(20, X_train.shape[1]),  # Select top 20 features, but not more than available
        classification=classification
    )
    
    # Keep only the selected features in the training, validation, and test sets
    X_train, X_val, X_test = X_train[selected_features], X_val[selected_features], X_test[selected_features]
    
    # Step 12: Initialize Model (ONLY ONCE, before training)
    if tune_hyperparameters:
        print(f"\n🔍 Tuning hyperparameters for {model_type}...")
        model, best_params = hyperparameter_tuning(X_train, y_train, model_type=model_type, cv=3)
        print(f" Best parameters found: {best_params}")
    else:
        model = get_model(model_type, classification=classification)
    
    # Step 13: Train & Evaluate Model
    if classification:
        model = train_and_evaluate_classifier(model, X_train, X_val, y_train, y_val, model_type=model_type)
    else:
        model = train_and_evaluate_random_forest(X_train, X_val, y_train, y_val, model_type=model_type)
        
    # Step 14: Final Evaluation on Test Set
    # Define category labels before evaluation
    y_test_pred = model.predict(X_test)

    if classification:
        if hasattr(model, "predict_proba"):
            y_test_pred = np.argmax(model.predict_proba(X_test), axis=1)  # Get class predictions
    
        #  Convert to integer class labels (Fixing the error)
        y_test = y_test.astype(int)
        y_test_pred = y_test_pred.astype(int)
        
        
        # Define category labels
        category_order = ["Very Low", "Low", "Medium", "High", "Very High"]
        category_labels = {i: label for i, label in enumerate(category_order)}
        
        # Get only classes that appear in `y_test` and `y_test_pred`
        unique_classes = np.unique(np.concatenate([y_test, y_test_pred]))
        filtered_category_order = [category_labels[i] for i in unique_classes]
        
        print(f"Final category labels used: {filtered_category_order}")  # Debugging
        
        # Generate the classification report with correct labels
        print("\n Classification Report:\n")
        # print_class_report(y_test, y_test_pred, labels=unique_classes, target_names=filtered_category_order)
        print_class_report(y_test, y_test_pred)
        
        #  Get only classes that appear in `y_test` and `y_test_pred`
        unique_classes = np.unique(np.concatenate([y_test, y_test_pred]))  
        filtered_category_order = [category_labels[i] for i in unique_classes]
    
        print(f"Final category labels used: {filtered_category_order}")  # Debugging
        
        #  Create filtered category labels
        filtered_category_order = [category_labels[i] for i in unique_classes]
    
        # Generate the confusion matrix with correct labels
        cm = confusion_matrix(y_test, y_test_pred, labels=unique_classes)
        
        # Plot confusion matrix with correct labels
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=filtered_category_order, 
                    yticklabels=filtered_category_order)
        
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title(f"Confusion Matrix for {model_type}")
        plt.show()

        #  Print Classification Report (fix: explicitly specify `labels=unique_classes`)
        print("\n Classification Report:\n")
        print_class_report(y_test, y_test_pred)
    
        #  Evaluate classification metricsprint_class_reportprint_class_report
        evaluate_with_confusion_matrix(y_test, y_test_pred, model_type=model_type, category_order=filtered_category_order)
        evaluate_with_classification_metrics(y_test, y_test_pred, model_type=model_type)
        
        # # Find the unique labels present in y_test
        # unique_classes = np.unique(y_test)
        
        # # Filter category_order dynamically based on present classes
        # filtered_category_order = [category_labels[i] for i in unique_classes]
        
        # # Print unique classes for debugging
        # print(f"Unique classes in y_test: {unique_classes}")
        # print(f"Filtered category order: {filtered_category_order}")
        
       
        evaluate_on_full_dataset(model, X_train, X_val, X_test, y_train, y_val, y_test, model_type=model_type, category_order=category_order)
 
    # Step 15: **Ranking Features (SHAP Feature Importance Analysis)**
    print("\n Feature Importance Analysis (SHAP, Permutation, or Model-based)...")
    visualize_shap_values(model, X_train, feature_names=X_train.columns, plot_type="summary", model_type=model_type)
    
    # Step 14: Final Evaluation on Test Set
    y_test_pred = model.predict(X_test)
    
    if classification:
        if hasattr(model, "predict_proba"):
            y_test_pred = np.argmax(model.predict_proba(X_test), axis=1)
    
        # Generate Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
    
        # Ensure labels match category codes correctly
        category_labels = {i: label for i, label in enumerate(category_order)}  # Map indices to categorical labels
    
        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=[category_labels[i] for i in range(len(category_order))], 
                    yticklabels=[category_labels[i] for i in range(len(category_order))])
        
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title(f"Confusion Matrix for {model_type}")
        plt.show()
    
        # Print Classification Report
        print("\n Classification Report:\n")
        print_class_report(y_test, y_test_pred)
    
        # Evaluate classification metrics
        evaluate_with_classification_metrics(y_test, y_test_pred, model_type=model_type)
    
    else:
        print("\nFinal Test Results:")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_test_pred)}")
        print(f"R-squared (R2) Score: {r2_score(y_test, y_test_pred)}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_test_pred)}")
    
    # Step 15: **Ranking Features (SHAP Feature Importance Analysis)**
    print("\n Feature Importance Analysis (SHAP, Permutation, or Model-based)...")
    visualize_shap_values(model, X_train, feature_names=X_train.columns, plot_type="summary", model_type=model_type)
    
    return df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model, y_test_pred, selected_features

def evaluate_on_full_dataset(model, X_train, X_val, X_test, y_train, y_val, y_test, model_type="model", category_order=None):
    """
    Evalueert het model op de volledige dataset (train, validatie en test) en maakt een confusion matrix.

    Parameters:
    - model: Getraind model
    - X_train, X_val, X_test (pd.DataFrame): Feature datasets
    - y_train, y_val, y_test (pd.Series): Doelvariabelen
    - model_type (str): Naam van het model (voor de grafiektitel)
    - category_order (list): Lijst met categorieën in de juiste volgorde

    Geeft:
    - Niets terug, maar toont een confusion matrix van de volledige dataset.
    """
    print("\n Evalueren van het model op de volledige dataset...")

    # Maak voorspellingen voor de volledige dataset
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Combineer alle datasets
    y_all = np.concatenate([y_train, y_val, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])

    # Zorg ervoor dat labels integer-gecodeerd zijn
    y_all = y_all.astype(int)
    y_all_pred = y_all_pred.astype(int)

    # Zorg dat de juiste labels worden gebruikt
    unique_classes = np.unique(np.concatenate([y_all, y_all_pred]))
    filtered_category_order = [category_order[i] for i in unique_classes] if category_order else sorted(unique_classes)

    # Genereer de confusion matrix
    cm = confusion_matrix(y_all, y_all_pred, labels=unique_classes)

    # Plot de confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=filtered_category_order, 
                yticklabels=filtered_category_order)

    plt.xlabel("Voorspelde labels")
    plt.ylabel("Werkelijke labels")
    plt.title(f"Confusion Matrix - Volledige Dataset ({model_type})")
    plt.show()

    # Print het classificatierapport
    print("\n Volledig Classificatierapport:\n")
    print(classification_report(y_all, y_all_pred, labels=unique_classes, target_names=filtered_category_order))

    plot_permutation_importance(model, X_test, y_test, model_type=model_type)

if __name__ == "__main__":
    print("Running models with different configurations:\n")

    # Run with Random Forest (tuned and untuned)
    print("Using RandomForestRegressor (Default):")
    df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model_rf, y_test_pred_rf, selected_features = main(
        model_type="random_forest", use_total_sleep=False, add_noise=False, tune_hyperparameters=True
    )

    # print("\nUsing RandomForestRegressor (Tuned):")
    # df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model_rf_tuned, y_test_pred_rf_tuned, selected_features = main(
    #     model_type="random_forest", use_total_sleep=True, add_noise=False, tune_hyperparameters=True
    # )

    # # Run with AdaBoost (tuned and untuned)
    # print("\nUsing AdaBoostRegressor (Default):")
    # df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model_ab, y_test_pred_ab, selected_features = main(
    #     model_type="ada_boost", use_total_sleep=False, add_noise=True, tune_hyperparameters=False
    # )

    # print("\nUsing AdaBoostRegressor (Tuned):")
    # df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model_ab_tuned, y_test_pred_ab_tuned, selected_features = main(
    #     model_type="ada_boost", use_total_sleep=False, add_noise=True, tune_hyperparameters=True
    # )

    # # Run with XGBoost
    # print("\nUsing XGBoost:")
    # df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model_xgb, y_test_pred_xgb, selected_features = main(
    #     model_type="xgboost", use_total_sleep=True, add_noise=False, tune_hyperparameters=False
    # )

    # # Run with SVM
    # print("\nUsing Support Vector Machine (SVM):")
    # df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model_svm, y_test_pred_svm, selected_features = main(
    #     model_type="svm", use_total_sleep=False, add_noise=False, tune_hyperparameters=False
    # )

    # # Run with Neural Networks
    # print("\nUsing Neural Network:")
    # df_raw, df_cleaned, df_no_outliers, df_normalized, df_non_sq, model_nn, y_test_pred_nn, selected_features = main(
    #     model_type="neural_network", use_total_sleep=True, add_noise=True, tune_hyperparameters=True
    # )
    
    #hilal