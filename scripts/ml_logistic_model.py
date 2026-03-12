# %% Import libraries
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve,
    average_precision_score,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


# %% Paths (for reproducibility when used as a standalone script)
BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "data"
outdir = BASE_DIR / "results" / "logistic_regression"
outdir.mkdir(parents=True, exist_ok=True)


# %% Load data
train_data = pd.read_csv(data_dir / "train_data_python.csv")
test_data = pd.read_csv(data_dir / "test_data_python.csv")

# Separate features and target for training data
X_train = train_data.drop(columns=['Evaluation'])
y_train = train_data['Evaluation']

# Separate features and target for testing data
X_test = test_data.drop(columns=['Evaluation'])
y_test = test_data['Evaluation']

# Drop "ID" and "patient_id" columns from training and testing datasets
X_train = X_train.drop(columns=['ID', 'patient_id'])
X_test = X_test.drop(columns=['ID', 'patient_id'])

# Set positive class for all metrics (Accept = 1)
POSITIVE_CLASS = 1  # Accept
NEGATIVE_CLASS = 0  # Reject

# %% Helper function to format numbers to 3 significant figures
def format_sig_fig(value, sig_fig=3):
    """
    Format a number to specified significant figures
    
    Parameters:
    -----------
    value : float or int
        Number to format
    sig_fig : int
        Number of significant figures (default: 3)
    
    Returns:
    --------
    str : Formatted string with exactly sig_fig significant figures
    """
    import math
    
    if value == 0 or (isinstance(value, float) and math.isnan(value)):
        return "0.00"
    
    # Handle negative numbers
    sign = -1 if value < 0 else 1
    abs_value = abs(value)
    
    # Calculate the order of magnitude
    if abs_value == 0:
        order = 0
    else:
        order = int(math.floor(math.log10(abs_value)))
    
    # Calculate the number of decimal places needed
    # For 3 sig figs: if order >= 0, use (sig_fig - order - 1) decimal places
    #                 if order < 0, use (sig_fig - order - 1) decimal places
    if order >= 0:
        # Number >= 1: e.g., 123.456 -> 123 (3 sig figs)
        decimal_places = max(0, sig_fig - order - 1)
    else:
        # Number < 1: e.g., 0.00123 -> 0.00123 (3 sig figs)
        decimal_places = sig_fig - order - 1
    
    # Format with calculated decimal places
    formatted = f"{value:.{decimal_places}f}"
    
    # Remove trailing zeros and decimal point if not needed
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    
    return formatted

# %% Define function to calculate all metrics
def calculate_metrics(y_true, y_pred, y_pred_proba, positive_class=None):
    """
    Calculate comprehensive metrics for binary classification
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    positive_class : int, str, or None
        Name/value of the positive class. If None, will be inferred from data.
        For numeric labels: 1 is assumed to be positive (Accept)
        For string labels: 'Accept' is assumed to be positive
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    # Detect label type and positive class
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if positive_class is None:
        # Infer positive class
        if 1 in unique_labels:
            positive_class = 1  # Numeric: 1 = Accept
        elif 'Accept' in unique_labels:
            positive_class = 'Accept'  # String: 'Accept'
        else:
            # Use the less frequent class as positive (typically minority class)
            label_counts = Counter(y_true)
            positive_class = min(label_counts.items(), key=lambda x: x[1])[0]
    
    # Determine negative class
    negative_class = [label for label in unique_labels if label != positive_class][0]
    
    # Ensure labels are in correct order: [negative_class, positive_class] = [Reject, Accept] = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[negative_class, positive_class])
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=positive_class)  # Recall/TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba, pos_label=positive_class)
    f1 = f1_score(y_true, y_pred, pos_label=positive_class)
    
    return {
        'Accuracy': accuracy,
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1,
        'AUROC': auc_roc,
        'AUPR': auc_pr
    }

# %% Shared hyperparameter grid and CV
param_grid_logreg = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear'],  # works with both l1 and l2
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=918)


# %% Strategy 1: Random Under-Sampling with 5-fold Cross-Validation
print("\n" + "="*70)
print("STRATEGY 1: Random Under-Sampling with 5-Fold Cross-Validation")
print("="*70)

# Create pipeline with undersampling and logistic regression
# Note: We need to use imblearn's Pipeline to ensure undersampling happens in each CV fold
# Scale first, then undersample, then classify
undersample_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('undersample', RandomUnderSampler(random_state=918, sampling_strategy='auto')),
    ('classifier', LogisticRegression(random_state=918, max_iter=1000))
])

# 5-fold cross-validation with grid search for hyperparameter tuning
grid_search_undersample = GridSearchCV(
    estimator=undersample_pipeline,
    param_grid=param_grid_logreg,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
) 

print("\nPerforming 5-fold cross-validation with grid search...")
print(f"Testing {len(param_grid_logreg['classifier__C']) * len(param_grid_logreg['classifier__penalty']) * len(param_grid_logreg['classifier__solver'])} parameter combinations...")
grid_search_undersample.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search_undersample.best_params_}")
print(f"Best CV score (AUC-ROC): {format_sig_fig(grid_search_undersample.best_score_)}")

# Get the best model (already trained on all training data with best parameters)
best_model_undersample = grid_search_undersample.best_estimator_

# Evaluate on test set
# Note: For prediction, we need to apply the same preprocessing pipeline
# The pipeline will handle scaling, but we need to ensure test data is processed correctly
y_pred_undersample = best_model_undersample.predict(X_test)
y_pred_proba_undersample = best_model_undersample.predict_proba(X_test)[:, 1]

# Calculate metrics (explicitly set positive_class=1 for Accept)
metrics_undersample = calculate_metrics(y_test, y_pred_undersample, y_pred_proba_undersample, positive_class=POSITIVE_CLASS)

print("\nTest Set Performance:")
print(f"Accuracy: {format_sig_fig(metrics_undersample['Accuracy'])}")
print(f"Sensitivity: {format_sig_fig(metrics_undersample['Sensitivity (Recall)'])}")
print(f"Specificity: {format_sig_fig(metrics_undersample['Specificity'])}")
print(f"F1 Score: {format_sig_fig(metrics_undersample['F1 Score'])}")
print(f"AUROC: {format_sig_fig(metrics_undersample['AUROC'])}")
print(f"AUPR: {format_sig_fig(metrics_undersample['AUPR'])}")

# %% Strategy 2: Class Weighting with 5-fold Cross-Validation
print("\n" + "="*70)
print("STRATEGY 2: Class Weighting with 5-Fold Cross-Validation")
print("="*70)

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

print(f"Computed class weights: {class_weight_dict}")

# Create pipeline with scaling and logistic regression (no undersampling)
classweight_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=918, max_iter=1000, class_weight='balanced'))
])

# 5-fold cross-validation with grid search for hyperparameter tuning
grid_search_classweight = GridSearchCV(
    estimator=classweight_pipeline,
    param_grid=param_grid_logreg,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
) 

print("\nPerforming 5-fold cross-validation with grid search...")
print(f"Testing {len(param_grid_logreg['classifier__C']) * len(param_grid_logreg['classifier__penalty']) * len(param_grid_logreg['classifier__solver'])} parameter combinations...")
grid_search_classweight.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search_classweight.best_params_}")
print(f"Best CV score (AUC-ROC): {format_sig_fig(grid_search_classweight.best_score_)}")

# Get the best model (already trained on all training data with best parameters)
best_model_classweight = grid_search_classweight.best_estimator_

# Evaluate on test set
y_pred_classweight = best_model_classweight.predict(X_test)
y_pred_proba_classweight = best_model_classweight.predict_proba(X_test)[:, 1]

# Calculate metrics (explicitly set positive_class=1 for Accept)
metrics_classweight = calculate_metrics(y_test, y_pred_classweight, y_pred_proba_classweight, positive_class=POSITIVE_CLASS)

print("\nTest Set Performance:")
print(f"Accuracy: {format_sig_fig(metrics_classweight['Accuracy'])}")
print(f"Sensitivity: {format_sig_fig(metrics_classweight['Sensitivity (Recall)'])}")
print(f"Specificity: {format_sig_fig(metrics_classweight['Specificity'])}")
print(f"F1 Score: {format_sig_fig(metrics_classweight['F1 Score'])}")
print(f"AUROC: {format_sig_fig(metrics_classweight['AUROC'])}")
print(f"AUPR: {format_sig_fig(metrics_classweight['AUPR'])}")

# %% Create comparison table
print("\n" + "="*70)
print("COMPARISON TABLE: Performance Summary")
print("="*70)

comparison_results = pd.DataFrame({
    'Strategy': ['Class Weighting', 'Random Under-Sampling'],
    'Accuracy': [metrics_classweight['Accuracy'], metrics_undersample['Accuracy']],
    'Sensitivity': [metrics_classweight['Sensitivity (Recall)'], metrics_undersample['Sensitivity (Recall)']],
    'Specificity': [metrics_classweight['Specificity'], metrics_undersample['Specificity']],
    'F1 Score': [metrics_classweight['F1 Score'], metrics_undersample['F1 Score']],
    'AUROC': [metrics_classweight['AUROC'], metrics_undersample['AUROC']],
    'AUPR': [metrics_classweight['AUPR'], metrics_undersample['AUPR']]
})

# Reorder columns: Strategy, Accuracy, Sensitivity, Specificity, F1 Score, AUROC, AUPR
comparison_results = comparison_results[['Strategy', 'Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'AUROC', 'AUPR']]

# Format the table for better readability (3 significant figures)
comparison_results_formatted = comparison_results.copy()
for col in ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'AUROC', 'AUPR']:
    comparison_results_formatted[col] = comparison_results_formatted[col].apply(lambda x: format_sig_fig(x))

print("\n" + comparison_results_formatted.to_string(index=False))

# Save comparison table
comparison_results.to_csv(outdir / "logistic_regression_comparison.csv", index=False)

print(f"\nComparison table saved to:")
print(f"  - {outdir / 'logistic_regression_comparison.csv'}")


