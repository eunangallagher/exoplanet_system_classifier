import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocessing import X_train, X_test, y_train, y_test

# Train a dummy classifier to establish a baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)
print("\nDummy Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
# confusion matrix
cm = confusion_matrix(y_test, y_pred_baseline)
print("Confusion Matrix:")
print(cm)
# print  true positives, false positives, false negatives, true negatives
tn, fp, fn, tp = cm.ravel()
print("True Positives:", tp)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Negatives:", tn)