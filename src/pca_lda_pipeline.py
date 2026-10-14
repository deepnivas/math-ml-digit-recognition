import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pca_from_scratch import PCAFromScratch
from lda_from_scratch import LDAFromScratch


# ============================================================
# LOAD DATA
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

print("=" * 60)
print("PCA + LDA + CLASSIFIER")
print("=" * 60)

print(f"Dataset shape : {X.shape}")
print(f"Number of classes : {len(np.unique(y))}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining data : {X_train.shape}")
print(f"Testing data  : {X_test.shape}")


# ============================================================
# PCA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: PCA")
print("=" * 60)

pca = PCAFromScratch(
    n_components=20
)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print(
    f"\nPCA training shape : {X_train_pca.shape}"
)

print(
    f"PCA testing shape  : {X_test_pca.shape}"
)

print(
    f"Variance retained  : "
    f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
)


# ============================================================
# LDA
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: LDA")
print("=" * 60)

lda = LDAFromScratch(
    n_components=9
)

X_train_lda = lda.fit_transform(
    X_train_pca,
    y_train
)

X_test_lda = lda.transform(
    X_test_pca
)

print(
    f"\nLDA training shape : {X_train_lda.shape}"
)

print(
    f"LDA testing shape  : {X_test_lda.shape}"
)


# ============================================================
# KNN CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: KNN CLASSIFIER")
print("=" * 60)

classifier = KNeighborsClassifier(
    n_neighbors=5
)

classifier.fit(
    X_train_lda,
    y_train
)

y_pred = classifier.predict(
    X_test_lda
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\n" + "=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)
