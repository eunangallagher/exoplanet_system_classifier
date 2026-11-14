from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score
)
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

# print classification metrics
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_baseline))
print("Precision (Multi-planet):", precision_score(y_test, y_pred_baseline))
print("Recall (Multi-planet):", recall_score(y_test, y_pred_baseline))
print("F1-Score (Multi-planet):", f1_score(y_test, y_pred_baseline))