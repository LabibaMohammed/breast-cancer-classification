
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

from common import load_data, split_data, get_models


# 1. Load dataset
X, y, data = load_data()


# 2. Split dataset
X_train, X_test, y_train, y_test = split_data(X, y)


# 3. Get models
models = get_models()


# 4. Train models and collect results
results = {}
roc_data = {}
confusion_matrices = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100

    sensitivity = recall_score(
        y_test,
        predictions,
        pos_label=0
    ) * 100

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    specificity = (
        cm[1, 1] / (cm[1, 1] + cm[1, 0])
    ) * 100

    results[name] = [
        accuracy,
        sensitivity,
        specificity
    ]

    # ROC curve for malignant class
    if hasattr(model, "decision_function"):
        scores = -model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)[:, 0]

    actual_malignant = (y_test == 0).astype(int)

    fpr, tpr, thresholds = roc_curve(
        actual_malignant,
        scores
    )

    roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    confusion_matrices[name] = cm


# 5. Plot performance comparison
metrics = ["Accuracy", "Sensitivity", "Specificity"]

for i, metric in enumerate(metrics):

    values = [
        results[name][i]
        for name in models
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(models.keys(), values)

    plt.title(metric + " Comparison")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        metric.lower() + "_comparison.png",
        dpi=300
    )

    plt.show()


# 6. Plot ROC curves
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, roc_auc) in roc_data.items():

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig("roc_comparison.png", dpi=300)
plt.show()


# 7. Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for ax, (name, cm) in zip(axes.flat, confusion_matrices.items()):

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Malignant", "Benign"]
    )

    display.plot(
        ax=ax,
        colorbar=False,
        values_format="d"
    )

    ax.set_title(name)

# Hide the unused sixth plot
axes.flat[5].axis("off")

plt.tight_layout()

plt.savefig("confusion_matrices.png", dpi=300)
plt.show()

print("All charts saved successfully!")
