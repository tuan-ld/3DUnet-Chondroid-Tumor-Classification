# train_classifiers.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from catboost import CatBoostClassifier

# Load and process data function
def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    train_df = df[df['Subset'] == 'train']
    test_df = df[df['Subset'] == 'test']
    
    train_numeric_cols = train_df.select_dtypes(include=['int', 'float']).columns
    test_numeric_cols = test_df.select_dtypes(include=['int', 'float']).columns
    
    numeric_train_df = train_df[train_numeric_cols]
    numeric_test_df = test_df[test_numeric_cols]
    
    X_train = numeric_train_df.drop(columns=['Class_Label'])
    y_train = numeric_train_df['Class_Label']
    
    X_test = numeric_test_df.drop(columns=['Class_Label'])
    y_test = numeric_test_df['Class_Label']
    
    return X_train, y_train, X_test, y_test

# Data scaling function
def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Apply SMOTE function
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled

# Compute confidence intervals function
def compute_ci(data, confidence=0.95):
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    margin_of_error = std_err * 1.96  # for 95% CI
    return mean - margin_of_error, mean + margin_of_error

# Classifier evaluation function with ROC plotting
def evaluate_classifiers(X_train_oversampled, y_train_oversampled, X_test_scaled, y_test, model_name, output_file):
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0)
    }
    
    results = []
    cv = StratifiedKFold(n_splits=5)
    
    for name, clf in classifiers.items():
        # Cross-validation on training data
        y_pred_cv = cross_val_predict(clf, X_train_oversampled, y_train_oversampled, cv=cv, method='predict')
        cv_accuracy = accuracy_score(y_train_oversampled, y_pred_cv)
        cv_weighted_kappa = cohen_kappa_score(y_train_oversampled, y_pred_cv, weights='quadratic')
        
        # Fit classifier and evaluate on test data
        clf.fit(X_train_oversampled, y_train_oversampled)
        test_predictions = clf.predict(X_test_scaled)
        
        # Metrics on test data
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_weighted_kappa = cohen_kappa_score(y_test, test_predictions, weights='quadratic')
        test_classification_report = classification_report(y_test, test_predictions)
        test_confusion_matrix = confusion_matrix(y_test, test_predictions)
        
        # ROC AUC calculation
        y_test_binary = pd.get_dummies(y_test)
        y_pred_proba = clf.predict_proba(X_test_scaled)
        roc_auc = roc_auc_score(y_test_binary, y_pred_proba, average='macro')

        # ROC Curve Plotting for each class
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(y_test_binary.columns):
            fpr, tpr, _ = roc_curve(y_test_binary.iloc[:, i], y_pred_proba[:, i])
            roc_auc_class = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc_class:.2f})")
        
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guess
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name} ({model_name})')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        # Compute confidence intervals for test accuracy using bootstrapping
        test_accuracy_list = []
        for _ in range(1000):  # Bootstrapping
            indices = np.random.choice(range(len(test_predictions)), size=len(test_predictions), replace=True)
            test_accuracy_list.append(accuracy_score(y_test.values[indices], test_predictions[indices]))

        ci_lower, ci_upper = compute_ci(test_accuracy_list)

        # Print results and store
        print(f"\nClassifier: {name}")
        print("Cross-Validated Accuracy (5-fold):", cv_accuracy)
        print("Cross-Validated Weighted Kappa (5-fold):", cv_weighted_kappa)
        print("Test Set Accuracy:", test_accuracy)
        print("Test Set Weighted Kappa:", test_weighted_kappa)
        print("Test Set AUC:", roc_auc)
        print("Classification Report (Test Set):")
        print(test_classification_report)
        print("Confusion Matrix (Test Set):")
        print(test_confusion_matrix)
        print(f"95% Confidence Interval for Test Accuracy: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        results.append({
            "Classifier": name,
            "Cross-Validated Accuracy (5-fold)": cv_accuracy,
            "Cross-Validated Weighted Kappa (5-fold)": cv_weighted_kappa,
            "Test Set Accuracy": test_accuracy,
            "Test Set Weighted Kappa": test_weighted_kappa,
            "Test Set AUC": roc_auc,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper
        })
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, 
                    xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
        plt.title(f'Confusion Matrix - {name} ({model_name})')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")
