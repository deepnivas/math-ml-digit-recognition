import os
import time
import joblib
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from src.pca_from_scratch import PCAFromScratch


print("=" * 60)
print("MNIST MODEL TRAINING")
print("=" * 60)

# Load the 28x28 MNIST dataset
print("\nLoading MNIST...")

digits = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X = np.asarray(digits.data, dtype=np.float32)
y = np.asarray(digits.target, dtype=np.int64)

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

# Use a stratified subset for a manageable first experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=12000,
    test_size=3000,
    random_state=42,
    stratify=y
)

print("Training images:", X_train.shape[0])
print("Testing images:", X_test.shape[0])

# PCA implemented in your own code
print("\nFitting PCA from scratch...")
start = time.time()

pca = PCAFromScratch(n_components=50)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA dimensions:", X_train_pca.shape[1])
print("PCA time:", round(time.time() - start, 2), "seconds")

# Experiment 1: PCA + KNN
print("\nTraining KNN...")
start = time.time()

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance"
)

knn.fit(X_train_pca, y_train)
knn_predictions = knn.predict(X_test_pca)

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)

print("KNN accuracy:", round(knn_accuracy * 100, 2), "%")
print("KNN time:", round(time.time() - start, 2), "seconds")

# Experiment 2: PCA + RBF SVM
print("\nTraining SVM...")
print("This step may take several minutes.")

start = time.time()

svm = SVC(
    kernel="rbf",
    C=5,
    gamma="scale",
    cache_size=1000,
    probability=True
)

svm.fit(X_train_pca, y_train)

svm_predictions = svm.predict(X_test_pca)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

print("SVM accuracy:", round(svm_accuracy * 100, 2), "%")
print("SVM time:", round(time.time() - start, 2), "seconds")

# Compare both classifiers
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"PCA + KNN : {knn_accuracy * 100:.2f}%")
print(f"PCA + SVM : {svm_accuracy * 100:.2f}%")

if svm_accuracy >= knn_accuracy:
    best_name = "PCA + SVM"
    best_model = svm
else:
    best_name = "PCA + KNN"
    best_model = knn

print("Best model:", best_name)

print("\nClassification report for the best model:")

if best_name == "PCA + SVM":
    best_predictions = svm_predictions
else:
    best_predictions = knn_predictions

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)

# Save the best model and PCA together
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "pca": pca,
        "model": best_model,
        "model_name": best_name,
        "accuracy": max(knn_accuracy, svm_accuracy)
    },
    "models/mnist_model.joblib"
)

print("Saved model: models/mnist_model.joblib")
print("\nTraining completed." )
