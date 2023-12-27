import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score

# Load the dataset
data = pd.read_csv('D:/11/train.csv')  # Replace 'your_dataset.csv' with the actual file path

# Extract features and labels
X = data.drop(['id', 'target'], axis=1)
y = data['target']

# Handling missing values if any
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
xgb_predictions = xgb_classifier.predict(X_test)
xgb_probabilities = xgb_classifier.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict(X_test)
gb_probabilities = gb_classifier.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_probabilities = dt_classifier.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# Evaluate XGBoost
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_auc = roc_auc_score(y_test, xgb_probabilities)
print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')
print(f'XGBoost AUC: {xgb_auc:.4f}')
print('XGBoost Classification Report:')
print(classification_report(y_test, xgb_predictions))

# Evaluate Gradient Boosting
gb_accuracy = accuracy_score(y_test, gb_predictions)
gb_auc = roc_auc_score(y_test, gb_probabilities)
print(f'Gradient Boosting Accuracy: {gb_accuracy:.4f}')
print(f'Gradient Boosting AUC: {gb_auc:.4f}')
print('Gradient Boosting Classification Report:')
print(classification_report(y_test, gb_predictions))

# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_auc = roc_auc_score(y_test, dt_probabilities)
print(f'Decision Tree Accuracy: {dt_accuracy:.4f}')
print(f'Decision Tree AUC: {dt_auc:.4f}')
print('Decision Tree Classification Report:')
print(classification_report(y_test, dt_predictions))

# Plot ROC curve for the three classifiers
plt.figure(figsize=(10, 8))
classifiers = {'XGBoost': xgb_probabilities, 'Gradient Boosting': gb_probabilities, 'Decision Tree': dt_probabilities}

for name, probs in classifiers.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], 'k--')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Show the plot
plt.show()

# Bar plot for model comparison
labels = ['XGBoost', 'Gradient Boosting', 'Decision Tree']
accuracies = [xgb_accuracy, gb_accuracy, dt_accuracy]
auc_scores = [xgb_auc, gb_auc, dt_auc]
f1_scores = [f1_score(y_test, xgb_predictions), f1_score(y_test, gb_predictions), f1_score(y_test, dt_predictions)]

# Plotting accuracy, AUC, and F1 score
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Accuracy comparison
axes[0].bar(labels, accuracies, color=['blue', 'orange', 'green'])
axes[0].set_ylim([0, 1])
axes[0].set_title('Accuracy Comparison')

# AUC comparison
axes[1].bar(labels, auc_scores, color=['blue', 'orange', 'green'])
axes[1].set_ylim([0, 1])
axes[1].set_title('AUC Comparison')

# F1 score comparison
axes[2].bar(labels, f1_scores, color=['blue', 'orange', 'green'])
axes[2].set_ylim([0, 1])
axes[2].set_title('F1 Score Comparison')

plt.show()
