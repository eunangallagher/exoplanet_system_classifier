from sklearn.utils import resample
import pandas as pd
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

# 1. Separate training data into single-planet (0) and multi-planet (1)
X_train_0 = X_train[y_train == 0]
y_train_0 = y_train[y_train == 0]
X_train_1 = X_train[y_train == 1]
y_train_1 = y_train[y_train == 1]

# 2. Oversample multi-planet rows in training set to match single-planet count
X_train_1_over, y_train_1_over = resample(
    X_train_1, y_train_1, replace=True, n_samples=len(X_train_0), random_state=42
)

# 3. Combine back into a balanced 50/50 training set
X_train_bal = pd.concat([X_train_0, X_train_1_over])
y_train_bal = pd.concat([y_train_0, y_train_1_over])

# 4. Train a simpler, regularized neural network on the balanced data
nn_pipeline = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=1000, random_state=42)
)

nn_pipeline.fit(X_train_bal, y_train_bal)
y_pred_nn_v2 = nn_pipeline.predict(X_test)

print("\nNeural Network (MLP) with Balanced Data Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn_v2))
cm_nn = confusion_matrix(y_test, y_pred_nn_v2)
print("Confusion Matrix:")
print(cm_nn)

tn_nn, fp_nn, fn_nn, tp_nn = cm_nn.ravel()
print("True Positives:", tp_nn)
print("False Positives:", fp_nn)
print("False Negatives:", fn_nn)
print("True Negatives:", tn_nn)
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_nn_v2))
print("Precision (Multi-planet):", precision_score(y_test, y_pred_nn_v2))
print("Recall (Multi-planet):", recall_score(y_test, y_pred_nn_v2))
print("F1-Score (Multi-planet):", f1_score(y_test, y_pred_nn_v2))