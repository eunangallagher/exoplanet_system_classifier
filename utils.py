import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names, filename="feature_importance.png"):
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances, color='steelblue')
    plt.xlabel("Importance Score")
    plt.title("Stellar Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_model_comparison(results_dict, filename="model_comparison.png"):
    models = list(results_dict.keys())
    metrics = ["Balanced Acc", "Precision", "Recall", "F1-Score"]
    
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        scores = [results_dict[m][i] for m in models]
        plt.bar(x + (i - 1.5) * width, scores, width, label=metric)

    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.ylabel("Score (0.0 to 1.0)")
    plt.title("Classifier Performance Comparison (Multi-Planet Class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")
