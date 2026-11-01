
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 55)
print("BREAST CANCER CLASSIFICATION")
print("=" * 55)

print("Total samples:", len(X))
print("Total features:", X.shape[1])


# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}


# 4. Repeated Cross-Validation setup
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# 5. Train and evaluate models
results = []

for name, model in models.items():

    print("\nTraining:", name)

    # Repeated cross-validation on training data
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    # Train on the full training set
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate test metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=0
    )

    recall = recall_score(
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
    malignant_scores = -model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        (y_test == 0).astype(int),
        malignant_scores
    )

    # Save results
    results.append({
        "Model": name,
        "Test Accuracy": accuracy * 100,
        "Malignant Precision": precision * 100,
        "Malignant Recall": recall * 100,
        "Malignant F1": f1 * 100,
        "Test ROC-AUC": roc_auc,
        "CV Accuracy Mean": cv_scores["test_accuracy"].mean() * 100,
        "CV Accuracy Std": cv_scores["test_accuracy"].std() * 100,
        "CV ROC-AUC Mean": cv_scores["test_roc_auc"].mean()
    })

    print("Test Accuracy:", round(accuracy * 100, 2), "%")
    print("Malignant Recall:", round(recall * 100, 2), "%")
    print("Test ROC-AUC:", round(roc_auc, 4))


# 6. Display comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV ROC-AUC Mean",
    ascending=False
)

print("\n" + "=" * 110)
print("MODEL COMPARISON")
print("=" * 110)

print(results_df.round(4).to_string(index=False))


# 7. Show detailed report for the best CV model
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

best_model.fit(X_train, y_train)

best_predictions = best_model.predict(X_test)

print("\nBest model based on CV ROC-AUC:", best_name)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_predictions,
        target_names=data.target_names
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)

print("\nExperiment completed!")
