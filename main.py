from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score
)
from preprocessing import X_train, X_test, y_train, y_test
from utils import plot_feature_importance, plot_model_comparison
from random_forest_classifier import rf_classifier, y_pred_rf
from decision_tree_classifier import y_pred_dt
from dummy_classifier import  y_pred_baseline
from neural_network_classifier import  y_pred_nn
from nn_classifier_v2 import  y_pred_nn_v2
from gradient_boost_classifier import  y_pred_hgb

# Save feature importance chart
plot_feature_importance(rf_classifier, X_train.columns)

# Collect scores for all 4 models: [Balanced Acc, Precision, Recall, F1]
results = {
    "Dummy": [
        balanced_accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline, zero_division=0),
        recall_score(y_test, y_pred_baseline, zero_division=0),
        f1_score(y_test, y_pred_baseline, zero_division=0)
    ],
    "Decision Tree": [
        balanced_accuracy_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_dt)
    ],
    "Random Forest": [
        balanced_accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf)
    ],
    "Raw Neural Net": [
        balanced_accuracy_score(y_test, y_pred_nn),
        precision_score(y_test, y_pred_nn, zero_division=0),
        recall_score(y_test, y_pred_nn, zero_division=0),
        f1_score(y_test, y_pred_nn, zero_division=0)
    ],
    "Balanced Neural Net": [
        balanced_accuracy_score(y_test, y_pred_nn_v2),
        precision_score(y_test, y_pred_nn_v2, zero_division=0),
        recall_score(y_test, y_pred_nn_v2, zero_division=0),
        f1_score(y_test, y_pred_nn_v2, zero_division=0)
    ],
    "Gradient Boosting": [
        balanced_accuracy_score(y_test, y_pred_hgb),
        precision_score(y_test, y_pred_hgb, zero_division=0),
        recall_score(y_test, y_pred_hgb, zero_division=0),
        f1_score(y_test, y_pred_hgb, zero_division=0)
    ]
}

plot_model_comparison(results)
