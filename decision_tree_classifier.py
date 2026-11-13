import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score
)
from sklearn.tree import DecisionTreeClassifier

from preprocessing import X_train, X_test, y_train, y_test

# train a decision tree classifier

dt_classifier = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
print("\nDecision Tree Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
# confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:")
print(cm_dt)
# print  true positives, false positives, false negatives, true negatives
tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()
print("True Positives:", tp_dt)
print("False Positives:", fp_dt)
print("False Negatives:", fn_dt)
print("True Negatives:", tn_dt)

# print classification metrics
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_dt))
print("Precision (Multi-planet):", precision_score(y_test, y_pred_dt))
print("Recall (Multi-planet):", recall_score(y_test, y_pred_dt))
print("F1-Score (Multi-planet):", f1_score(y_test, y_pred_dt))