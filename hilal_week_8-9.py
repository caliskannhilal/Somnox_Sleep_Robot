

"""
Created on Thu Sep  5 13:37:59 2024
@author: hilalcaliskan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from sklearn.ensemble import RandomForestRegressor


# Load the dataset
df = pd.read_csv('/Users/hilalcaliskan/Documents/Python_work_directory/Diplora progress/Somnox/Living_Somnox_feature_df_130_[2024, 5, 1].csv')

## PART 1 - Data Pre-processing ##

# 1. Data inspection
print("Head of the Dataframe:")
print(df.head())

print("Info of the DataFrame:")
df.info()

# Checking for missing values in the entire DataFrame
print("Missing values per column:")
print(df.isnull().sum())

# Print the list of column names
column_list = df.columns.tolist()
print(column_list)

# Choose your target column (replace with the actual column name)
target_column = 'f_SQ_Fitbit_details_SleepEfficiency'

# Inspect the target column
print(df[target_column].describe())  # Summary statistics
print(df[target_column].isnull().sum())  # Check for missing values

# Visualize the distribution of the target variable
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(df[target_column], bins=20, kde=True)
plt.title(f'Distribution of {target_column}')
plt.show()

# Boxplot for outliers detection
plt.figure(figsize=(8, 6))
sns.boxplot(x=df[target_column])
plt.title(f'Boxplot of {target_column}')
plt.show()

# 2. Cleaning the Dataset

# Step 1: Drop columns with more than 10% missing values
threshold_missing = 0.1  # 10% threshold for missing values
columns_to_drop_missing = df.columns[df.isnull().mean() > threshold_missing]
df_cleaned = df.drop(columns=columns_to_drop_missing)

# Step 2: Fill missing values for numeric columns
for col in df_cleaned.columns:
    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

# Step 3: Drop columns with more than 90% zero values
threshold_zeros = 0.9  # 90% threshold for zero values
columns_to_drop_zeros = df_cleaned.columns[(df_cleaned == 0).mean() > threshold_zeros]
df_cleaned.drop(columns=columns_to_drop_zeros, inplace=True)

# Exclude non-numeric columns like the subject column (assuming it's named 's_data subject')
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns  # Select only numeric columns

############# Scaling ##################

# Step 4: Apply MinMax scaling only to numeric columns
scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[numeric_columns]), columns=numeric_columns)

# Verify scaling
print("Data after MinMax scaling:")
print(df_scaled.describe())

## PART 2 - Outlier Handling ##
# Set the Z-score threshold
threshold_z = 4  # 4 standard deviations away from the mean
outlier_percentage_threshold = 0.1  # 10% threshold for columns with too many outliers

# Calculate Z-scores for all numeric columns
z_scores = np.abs(stats.zscore(df_scaled))

# Identify columns with more than the threshold percentage of outliers
columns_to_drop_due_to_outliers = []
for i, col in enumerate(df_scaled.columns):
    outliers_in_column = (z_scores[:, i] > threshold_z).mean()  # Calculate % of outliers in this column
    if outliers_in_column > outlier_percentage_threshold:
        columns_to_drop_due_to_outliers.append(col)

print(f"Columns to drop due to too many outliers: {columns_to_drop_due_to_outliers}")

# Drop these columns with too many outliers
df_cleaned_outliers_columns_removed = df_scaled.drop(columns=columns_to_drop_due_to_outliers)

# Recalculate Z-scores for the remaining columns
z_scores_cleaned = np.abs(stats.zscore(df_cleaned_outliers_columns_removed))

# Identify rows where any value's Z-score is greater than the threshold
outliers_mask_cleaned = (z_scores_cleaned > threshold_z).any(axis=1)

# Drop the rows with extreme outliers
df_no_outliers = df_cleaned_outliers_columns_removed[~outliers_mask_cleaned]

# Verify the shape of the new DataFrame
print(f"Shape of the dataset after dropping extreme outliers: {df_no_outliers.shape}")


###### RFE #######

# Define the target column ( 'f_SQ_Fitbit_details_SleepEfficiency')
target_column = 'f_SQ_Fitbit_details_SleepEfficiency'

# Use the scaled dataset (df_scaled) for feature selection via RFE
# Assuming df_scaled contains all the cleaned and scaled SQ features

# Split the data into features (X) and target (y)
X = df_scaled.drop(columns=[target_column])
y = df_scaled[target_column]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a RandomForest model for RFE (using Regressor instead of Classifier)
rf_model = RandomForestRegressor(random_state=42, n_estimators=20)  # Reduce the number of trees (n_estimators)

# Apply RFE to select the top features (set n_features_to_select as needed)
rfe = RFE(estimator=rf_model, n_features_to_select=5)  # Reduce to 5 features (adjust as needed)
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe.support_]
print(f"Selected features: {selected_features}")

# Transform the data to only use the selected features
X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

# Now you are ready to train the model with the selected features
rf_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_selected)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R-squared: {r2}, MSE: {mse}")

# Get feature importance from the trained RandomForest model
feature_importances = rf_model.feature_importances_

# Display feature importances
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Plot feature importance
importance_df.plot(kind='bar', x='Feature', y='Importance', title='Feature Importance')
plt.show()

######## T-sne #########

# Using `df_scaled` because it is preprocessed and scaled DataFrame excluding the target column
X_tsne = df_scaled.drop(columns=['f_SQ_Fitbit_details_SleepEfficiency'])  # Exclude target column for clustering

# Initialize K-Means clustering (assuming 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit K-Means to the scaled data
kmeans.fit(X_tsne)

# Get the cluster labels for each point
df_scaled['cluster'] = kmeans.labels_

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)

# Fit t-SNE on the original high-dimensional features (X_tsne)
X_tsne_result = tsne.fit_transform(X_tsne)  # X_tsne is the original features excluding the target

# Plot the t-SNE result colored by K-Means clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne_result[:, 0], X_tsne_result[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(label='Cluster')
plt.title('t-SNE Visualization Colored by K-Means Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Check the cluster assignments
print(df_scaled[['cluster']].value_counts())