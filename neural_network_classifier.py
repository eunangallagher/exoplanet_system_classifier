
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from preprocessing import X_train, X_test, y_train, y_test

# Build a pipeline that scales features first, then trains the neural network
nn_pipeline = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
)

nn_pipeline.fit(X_train, y_train)
y_pred_nn = nn_pipeline.predict(X_test)

print("\nNeural Network (MLP) Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
cm_nn = confusion_matrix(y_test, y_pred_nn)
print("Confusion Matrix:")
print(cm_nn)

tn_nn, fp_nn, fn_nn, tp_nn = cm_nn.ravel()
print("True Positives:", tp_nn)
print("False Positives:", fp_nn)
print("False Negatives:", fn_nn)
print("True Negatives:", tn_nn)
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_nn))
print("Precision (Multi-planet):", precision_score(y_test, y_pred_nn))
print("Recall (Multi-planet):", recall_score(y_test, y_pred_nn))
print("F1-Score (Multi-planet):", f1_score(y_test, y_pred_nn))