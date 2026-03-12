from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

# %% Paths (for reproducibility when used as a standalone script)
BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "data"
outdir = BASE_DIR / "results" / "randomforest"
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


# Set positive/negative class labels (for metrics)
POSITIVE_CLASS = 1  # Accept
NEGATIVE_CLASS = 0  # Reject


def format_sig_fig(value, sig_fig=3):
    """
    Format a number to specified significant figures.
    """
    import math

    if value == 0 or (isinstance(value, float) and math.isnan(value)):
        return "0.00"

    abs_value = abs(value)
    if abs_value == 0:
        order = 0
    else:
        order = int(math.floor(math.log10(abs_value)))

    if order >= 0:
        decimal_places = max(0, sig_fig - order - 1)
    else:
        decimal_places = sig_fig - order - 1

    formatted = f"{value:.{decimal_places}f}"
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted


def calculate_metrics(y_true, y_pred, y_pred_proba, positive_class=None):
    """
    Calculate core binary classification metrics (matching logistic script).
    """
    if positive_class is None:
        positive_class = POSITIVE_CLASS

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    negative_class = [label for label in unique_labels if label != positive_class][0]

    cm = confusion_matrix(y_true, y_pred, labels=[negative_class, positive_class])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=positive_class)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba, pos_label=positive_class)
    f1 = f1_score(y_true, y_pred, pos_label=positive_class)

    return {
        'Accuracy': accuracy,
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1,
        'AUROC': auc_roc,
        'AUPR': auc_pr,
    }


## Random Forest strategies with class weighting vs down-sampling

# Shared RF hyperparameter grid
param_grid_rf = {
    'max_features': list(range(1, X_train.shape[1] + 1)),  # 1..n_features
    'n_estimators': list(range(1, 5001, 50)),              # 1..5000 step 50
}


def run_grid_search(strategy_name, base_estimator, X_train, y_train, X_test, y_test):
    """
    Run GridSearchCV for a given RF strategy and evaluate on the test set.
    """
    print("\n" + "=" * 70)
    print(strategy_name.upper())
    print("=" * 70)

    n_combinations = len(param_grid_rf['max_features']) * len(param_grid_rf['n_estimators'])
    print("\nPerforming grid search...")
    print(f"Testing {n_combinations} parameter combinations...")

    grid = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid_rf,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV score (AUC-ROC): {format_sig_fig(grid.best_score_)}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, positive_class=POSITIVE_CLASS)

    print("\nTest Set Performance:")
    print(f"Accuracy: {format_sig_fig(metrics['Accuracy'])}")
    print(f"Sensitivity: {format_sig_fig(metrics['Sensitivity (Recall)'])}")
    print(f"Specificity: {format_sig_fig(metrics['Specificity'])}")
    print(f"F1 Score: {format_sig_fig(metrics['F1 Score'])}")
    print(f"AUROC: {format_sig_fig(metrics['AUROC'])}")
    print(f"AUPR: {format_sig_fig(metrics['AUPR'])}")

    return best_model, metrics



# %% RF with down-sampling
# Get the total number of CPU cores
#total_cores = os.cpu_count()

# Use half of the available cores
#n_jobs_to_use = max(1, total_cores // 2)  # Ensure at least 1 core is used
#print(f"Total cores available: {total_cores}, using {n_jobs_to_use} cores.")

rf = BalancedRandomForestClassifier(
    oob_score=True,
    n_estimators=100, # Default number of trees for the model during initialization, but it gets overridden in GridSearchCV
    random_state=918, # Set random seed for reproducibility
    n_jobs=-1, # use half of the cores (to use all available cores set n_jobs=-1)
    replacement=True,
    bootstrap=True  # Set bootstrap=True for OOB estimation
)

# Define search space for max_features (mtry) and n_estimators (ntree)
param_grid = {
    'max_features': list(range(1, 72)),  # mtry from 1 to 72 since there are 72 features in total
    'n_estimators': list(range(1, 5001, 50))  # ntree from 1 to 5000 with 50 interval
}

# Grid search with OOB scoring (not used directly, but available after fit)
grid_search = GridSearchCV(
    estimator=rf,               # The base model to tune (BalancedRandomForestClassifier in this case)
    param_grid=param_grid,      # Dictionary of hyperparameters to search over
    scoring='roc_auc',         # Metric to evaluate model performance (could choose "accuracy, "f1", "roc_auc", etc.)
    cv=3,                       # Number of cross-validation folds (3-fold cross-validation)
    n_jobs=-1,       # Number of CPU cores to use for parallel processing
    verbose=2                   # Level of verbosity (2 provides detailed output during the search)
)


# Use grid search to fit the model with encoded data, find the best parameters
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_mtry = best_params['max_features']
best_ntree = best_params['n_estimators']
print(f"Best max_features (mtry): {best_mtry}")
print(f"Best n_estimators (ntree): {best_ntree}")
print(f"Best parameters: {best_params}")

# Save the best mtry and ntree
joblib.dump(best_mtry, outdir / "best_mtry_rf.pkl")
joblib.dump(best_ntree, outdir / "best_ntree_rf.pkl")

# Save the GridSearchCV results to a CSV file
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv(outdir / "grid_search_results.csv", index=False)
print("GridSearchCV results saved to grid_search_results.csv")

# Save the best parameters to a text file
with open(outdir / "best_params.txt", "w") as f:
    f.write(f"Best max_features (mtry): {best_mtry}\n")
    f.write(f"Best n_estimators (ntree): {best_ntree}\n")
    f.write(f"Best parameters: {best_params}\n")
print("Best parameters saved to best_params.txt")

# Save the trained model
joblib.dump(rf, outdir / "rf_tune_model.pkl")

# Train the final Random Forest model
rf_ds = BalancedRandomForestClassifier(
    n_estimators=best_ntree,
    max_features=best_mtry,
    oob_score=True, 
    random_state=918,
    n_jobs=-1,
    replacement=True,
    bootstrap=True  # Set bootstrap=True for OOB estimation
)
rf_ds.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_ds, outdir / "rf_downsample_model.pkl")
joblib.dump(rf_ds, project_dir / "output_python" / "model_artifacts" / "rf_downsample_model.pkl")


# OOB
# Evaluate the model
oob_error = 1 - rf_ds.oob_score_
print(f"OOB Error: {oob_error}") # the proportion of misclassified samples among the OOB samples.

# extract the OOB predictions
oob_predictions = rf_ds.oob_decision_function_[:, 1] > 0.5 
# each row corresponds to a sample and each column corresponds to the class probabilities. 
# The first column ([:, 0]) gives the probability of the negative class ("Reject" in our case since it is mapped to 0).
# The second column ([:, 1]) gives the probability of the positive class ("Accept" in our case since it is mapped to 1).
# "True" for predicted as "Reject"

# Convert boolean predictions to integers
oob_predictions_int = oob_predictions.astype(int) # change True to 1 and False to 0 (0: Reject, 1: Accept)

# Calculate the confusion matrix
oob_conf_matrix = confusion_matrix(y_train, oob_predictions_int)
# The confusion matrix is a 2x2 matrix with the following structure:
# [[TN, FP],
# [FN, TP]]

print("OOB Confusion Matrix:")
print(oob_conf_matrix)

# Extract TN, FP, FN, TP from the confusion matrix
TN, FP, FN, TP = oob_conf_matrix.ravel()

# Calculate sensitivity and specificity
oob_sensitivity = TP / (TP + FN)
oob_specificity = TN / (TN + FP)

print(f"Sensitivity (Recall or True Positive Rate): {oob_sensitivity}")
print(f"Specificity (True Negative Rate): {oob_specificity}")

# Plot the confusion matrix with labels
plt.figure(figsize=(7, 5))
sns.heatmap(oob_conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Reject', 'Predicted Accept'], # column names
            yticklabels=['Actual Reject', 'Actual Accept']) # row names
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Python OOB Confusion Matrix')
plt.savefig(outdir / "oob_conf_matrix.png")
plt.show()

# Calculate OOB ROC AUC
rf_ds_oob_roc_auc = roc_auc_score(y_train, rf_ds.oob_decision_function_[:, 1])
print(f"OOB ROC AUC: {rf_ds_oob_roc_auc}")

# Plot OOB ROC curve
fpr, tpr, _ = roc_curve(y_train, rf_ds.oob_decision_function_[:, 1])
plt.plot(fpr, tpr, label=f'OOB ROC curve (area = {rf_ds_oob_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Python OOB ROC Curve')
plt.legend(loc='best')
plt.savefig(outdir / "oob_roc_curve.png")
plt.show()



# Test set evaluation 
# Predict on test set
rf_ds_pred = rf_ds.predict(X_test)

rf_ds_conf_matrix = confusion_matrix(y_test, rf_ds_pred)
print("Test Set Confusion Matrix:")
print(rf_ds_conf_matrix)

rf_ds_pred_accuracy = accuracy_score(y_test, rf_ds_pred)
print(f"Test Set Accuracy: {rf_ds_pred_accuracy}")

# Extract TN, FP, FN, TP from the confusion matrix and convert to standard Python integers
TN, FP, FN, TP = [int(x) for x in rf_ds_conf_matrix.ravel()]

# Calculate sensitivity and specificity
rf_ds_sensitivity = TP / (TP + FN)
rf_ds_specificity = TN / (TN + FP)

print(f"Sensitivity (Recall or True Positive Rate): {rf_ds_sensitivity}")
print(f"Specificity (True Negative Rate): {rf_ds_specificity}")


rf_ds_pred_prob = rf_ds.predict_proba(X_test)[:, 1]
rf_ds_roc_auc = roc_auc_score(y_test, rf_ds_pred_prob)
print(f"Test Set ROC AUC: {rf_ds_roc_auc}")

# Calculate F1 score
rf_ds_f1 = f1_score(y_test, rf_ds_pred)
print(f"Test Set F1 Score: {rf_ds_f1}")

plt.figure(figsize=(10, 8))
sns.heatmap(rf_ds_conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Reject', 'Predicted Accept'], # column names
            yticklabels=['Actual Reject', 'Actual Accept'], # row names
            annot_kws={'size': 16})  # Make numbers larger
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(fontsize=16)
#plt.title('Random Forest with Down-samping Test set Confusion Matrix')
plt.savefig(outdir / "rf_ds_test_conf_matrix.png")
plt.show()

#plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, rf_ds_pred_prob)
plt.plot(fpr, tpr, label=f'Test Set ROC curve (AUC = {rf_ds_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Test Set ROC Curve')
plt.legend(loc='best')
plt.savefig(outdir / "rf_ds_test_set_roc_curve.png")
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score
# For test set predictions
test_precision, test_recall, test_thresholds = precision_recall_curve(y_test, rf_ds.predict_proba(X_test)[:, 1])
rf_ds_auc_pr = average_precision_score(y_test, rf_ds.predict_proba(X_test)[:, 1])

print(f"Test Set AUC-PR (down-sampling): {rf_ds_auc_pr}")
#plt.figure(figsize=(8, 6))
plt.plot(test_recall, test_precision, label=f'Test Set PR curve (AP = {rf_ds_auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Test Set Precision-Recall Curve')
plt.legend()
plt.savefig(outdir / "rf_ds_test_set_pr_curve.png")
#plt.grid(True)
plt.show()


# %% RF with down-sampling variable of importance
# Variable importance
importances = rf_ds.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_train.columns

# Export a CSV file with feature importance, feature name, and importance score. Order by most important at top.
feature_importance_df = pd.DataFrame({
    "Feature": features[indices],
    "Importance": importances[indices]
})
feature_importance_df.to_csv(outdir / "rf_ds_feature_importance.csv", index=False)


# 
plt.figure(figsize=(10, 8))
#plt.title("Random Forest with Down-Sampling Variable Importance (Top 15)", fontsize=20)
plt.barh(range(15), importances[indices[:15]], align="center")
plt.yticks(range(15), features[indices[:15]], fontsize=16)  # Increase y-tick label size
plt.xticks(fontsize=16)  # Increase x-tick label size
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.xlabel('Importance', fontsize=18)
plt.ylabel('Features', fontsize=18)
plt.tight_layout()
plt.savefig(outdir / "rf_ds_variable_importance_top_15.png")
plt.show()