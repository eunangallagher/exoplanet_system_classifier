from sklearn.ensemble import HistGradientBoostingClassifier
from preprocessing import X_train, X_test, y_train, y_test
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score
)

# Train HistGradientBoosting with balanced class weights and shallow trees to prevent overfitting
hgb_classifier = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=4,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42
)

hgb_classifier.fit(X_train, y_train)
y_pred_hgb = hgb_classifier.predict(X_test)

print("\nHistGradientBoosting Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_hgb))
cm_hgb = confusion_matrix(y_test, y_pred_hgb)
print("Confusion Matrix:")
print(cm_hgb)

tn_hgb, fp_hgb, fn_hgb, tp_hgb = cm_hgb.ravel()
print("True Positives:", tp_hgb)
print("False Positives:", fp_hgb)
print("False Negatives:", fn_hgb)
print("True Negatives:", tn_hgb)
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_hgb))
print("Precision (Multi-planet):", precision_score(y_test, y_pred_hgb))
print("Recall (Multi-planet):", recall_score(y_test, y_pred_hgb))
print("F1-Score (Multi-planet):", f1_score(y_test, y_pred_hgb))
