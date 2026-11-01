
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from common import load_data, split_data, get_models


# 1. Load and split the dataset
X, y, data = load_data()

X_train, X_test, y_train, y_test = split_data(X, y)


# 2. Get models
models = get_models()

results = []


# 3. Train models and calculate statistics
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tp, fn, fp, tn = cm.ravel()

    accuracy = accuracy_score(y_test, predictions)

    sensitivity = recall_score(
        y_test,
        predictions,
        pos_label=0
    )

    specificity = tn / (tn + fp)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=0
    )

    # Calculate ROC-AUC for malignant class
    if hasattr(model, "decision_function"):
        scores = -model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)[:, 0]

    actual = (y_test == 0).astype(int)

    roc_auc = roc_auc_score(actual, scores)

    results.append({
        "Model": name,
        "Accuracy (%)": accuracy * 100,
        "Sensitivity (%)": sensitivity * 100,
        "Specificity (%)": specificity * 100,
        "Precision (%)": precision * 100,
        "F1-score (%)": f1 * 100,
        "ROC-AUC": roc_auc,
        "False Negatives": fn,
        "False Positives": fp
    })


# 4. Display results
results_df = pd.DataFrame(results)

print("\nMODEL STATISTICS")
print(results_df.to_string(index=False))
