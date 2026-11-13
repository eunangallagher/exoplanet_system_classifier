import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score
)

from sklearn.ensemble import RandomForestClassifier

from preprocessing import X_train, X_test, y_train, y_test

# train a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print("\nRandom Forest Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
# confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(cm_rf)
# print  true positives, false positives, false negatives, true negatives
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
print("True Positives:", tp_rf)
print("False Positives:", fp_rf)
print("False Negatives:", fn_rf)
print("True Negatives:", tn_rf)

# print classification metrics
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_rf))
print("Precision (Multi-planet):", precision_score(y_test, y_pred_rf))
print("Recall (Multi-planet):", recall_score(y_test, y_pred_rf))
print("F1-Score (Multi-planet):", f1_score(y_test, y_pred_rf))