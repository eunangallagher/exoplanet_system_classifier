# Milestone 3: Model Comparison Checkpoint
**Student Name:** Eunan Ó Gallachóir  
**Student ID:** 23437944  
**Project Repository:** https://github.com/eunangallagher/exoplanet_system_classifier

---

## 1. Model Comparison Results Table
All models were trained on the exact same stratified 80% training split (2,674 systems) and evaluated on the 20% test set (669 systems).

| Model | Overrall Accuracy | Balanced Accuracy | Precision (Multi) | Recall (Multi) | F1-Score (Multi) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 82.36% | 50.00% | 0.00 | 0.00 | 0.00 |
| **Decision Tree Classifier** | 49.33% | 58.58% | 0.22 | 0.73 | 0.34 |
| **Random Forest Classifier** | 59.34% | 57.00% | 0.23 | 0.53 | 0.32 |
| **Raw Neural Net (MLP v1)** | 81.91% | 49.73% | 0.00 | 0.00 | 0.00 |
| **Balanced Neural Net (MLP v2)** | 60.84% | 56.58% | 0.23 | 0.50 | 0.31 |

---

## 2. Main Model Settings & Tuning Explanation
* **Tree Class Balancing (`class_weight="balanced"`):** Because single-planet systems make up 82.4% of the dataset standard models ignore the minority class. SSetting `class_weight="balanced"` in the Decision Tree (`max_depth=5`) and Random Forest (`n_estimators=100`, `max_depth=8`) penalized errors on multi-planet systems more heavily and prevented overfitting.
*   **Neural Network Scaling & Oversampling:** Features were scaled using `StandardScaler` inside a pipeline to prevent data leakage and stop large temperature values (`st_teff`) from dominating smaller metallicity values (`st_met`). Because `MLPClassifier` lacks a `class_weight` parameter, `MLP v1` collapsed into guessing the majority class. For `MLP v2`, random oversampling (`sklearn.utils.resample`) was applied strictly to `X_train` to create a 50/50 balanced training set, paired with smaller hidden layers `(32, 16)` and L2 regularization (`alpha=0.1`).

---
## 3. Visualisations & Error Analysis
Two visualisations were generated via `utils.py` to evaluate the models:
*   **`model_comparison.png`:** Compares Balanced Accuracy, Precision, Recall, and F1-Score across all five models, clearly illustrating the jump from `0.00` Recall on the Dummy and Raw Neural Net models to balanced detection on the tuned models.
*   **`feature_importance.png`:** Shows that **Effective Temperature (`st_teff`, ~0.24)** and **Surface Gravity (`st_logg`, ~0.22)** were the strongest predictors in the Random Forest, while **Number of Stars (`sy_snum`, ~0.015)** contributed the least.

### Where the Models Perform Poorly (Error Analysis)
*   **False Positives on Single-Planet Stars:** Forcing the models to detect the minority class caused them to over-predict multi-planet systems. The Decision Tree falsely flagged **307** single-planet stars as multi-planet systems, while the Random Forest reduced this to **217**, and the Balanced Neural Net achieved the lowest false-alarm count at **203**.
*   **Missed Multi-Planet Systems (False Negatives):** While the Balanced Neural Net kept false alarms lowest, it still missed half of the true multi-planet systems in the test set (`59` True Positives vs. `59` False Negatives). In contrast, the Decision Tree caught `86` of `118` multi-planet systems (`0.73` Recall) but suffered from poor overall accuracy (`49.33%`).

---

## 4. Draft Conclusion & Next Steps
*   **Current Best Model:** The **Balanced Neural Network (MLP v2)** is currently the best overall model. Among all models that actually detect multi-planet systems, it achieved the highest overall accuracy (**60.84%**) and the lowest number of false positives (**203**), while maintaining a solid multi-planet Recall (**0.50**) and F1-Score (**0.31**). The **Random Forest** is a very close runner-up (`59.34%` accuracy, `57.00%` balanced accuracy) and provides valuable feature interpretability.
*   **Key Limitation:** Many "single-planet" stars in the NASA archive likely host additional smaller or wider-orbiting planets that current telescopes have not yet detected (selection bias). This introduces noisy labels into the dataset where true multi-planet stars are mislabelled as Class 0.
*   **Next Steps for Milestone 4:** Test hyperparameter tuning (such as testing different probability thresholds or neural network layer sizes), generate confusion matrix visualisations for the final report, and prepare the 3,000-word report and 5-minute demonstration video.

---

## 5. Assistance Declaration
I used GitHub Copilot in VS Code to help structure the scikit-learn evaluation metrics, oversampling pipeline and matplotlib plotting syntax. All code was run, verified, and interpreted by me locally.