import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pca_from_scratch import PCAFromScratch
from lda_from_scratch import LDAFromScratch


# ============================================================
# LOAD DATA
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# FUNCTION TO TRAIN AND EVALUATE KNN
# ============================================================

def evaluate_model(name, X_train_data, X_test_data):

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(
        X_train_data,
        y_train
    )

    predictions = model.predict(
        X_test_data
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    return accuracy, precision, recall, f1


# ============================================================
# EXPERIMENT 1: RAW PIXELS
# ============================================================

print("=" * 70)
print("EXPERIMENT 1: RAW PIXELS + KNN")
print("=" * 70)

raw_results = evaluate_model(
    "Raw Pixels",
    X_train,
    X_test
)

print(
    f"Features  : {X_train.shape[1]}"
)

print(
    f"Accuracy  : {raw_results[0] * 100:.2f}%"
)

print(
    f"Precision : {raw_results[1] * 100:.2f}%"
)

print(
    f"Recall    : {raw_results[2] * 100:.2f}%"
)

print(
    f"F1 Score  : {raw_results[3] * 100:.2f}%"
)


# ============================================================
# EXPERIMENT 2: PCA + KNN
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: PCA + KNN")
print("=" * 70)

pca = PCAFromScratch(
    n_components=20
)

X_train_pca = pca.fit_transform(
    X_train
)

X_test_pca = pca.transform(
    X_test
)

pca_results = evaluate_model(
    "PCA",
    X_train_pca,
    X_test_pca
)

print(
    f"Features  : {X_train_pca.shape[1]}"
)

print(
    f"Variance  : "
    f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
)

print(
    f"Accuracy  : {pca_results[0] * 100:.2f}%"
)

print(
    f"Precision : {pca_results[1] * 100:.2f}%"
)

print(
    f"Recall    : {pca_results[2] * 100:.2f}%"
)

print(
    f"F1 Score  : {pca_results[3] * 100:.2f}%"
)


# ============================================================
# EXPERIMENT 3: PCA + LDA + KNN
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 3: PCA + LDA + KNN")
print("=" * 70)

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

lda_results = evaluate_model(
    "PCA + LDA",
    X_train_lda,
    X_test_lda
)

print(
    f"Features  : {X_train_lda.shape[1]}"
)

print(
    f"Accuracy  : {lda_results[0] * 100:.2f}%"
)

print(
    f"Precision : {lda_results[1] * 100:.2f}%"
)

print(
    f"Recall    : {lda_results[2] * 100:.2f}%"
)

print(
    f"F1 Score  : {lda_results[3] * 100:.2f}%"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"{'Method':<25}"
    f"{'Features':>10}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print("-" * 83)

print(
    f"{'Raw Pixels':<25}"
    f"{64:>10}"
    f"{raw_results[0] * 100:>11.2f}%"
    f"{raw_results[1] * 100:>11.2f}%"
    f"{raw_results[2] * 100:>11.2f}%"
    f"{raw_results[3] * 100:>11.2f}%"
)

print(
    f"{'PCA':<25}"
    f"{20:>10}"
    f"{pca_results[0] * 100:>11.2f}%"
    f"{pca_results[1] * 100:>11.2f}%"
    f"{pca_results[2] * 100:>11.2f}%"
    f"{pca_results[3] * 100:>11.2f}%"
)

print(
    f"{'PCA + LDA':<25}"
    f"{9:>10}"
    f"{lda_results[0] * 100:>11.2f}%"
    f"{lda_results[1] * 100:>11.2f}%"
    f"{lda_results[2] * 100:>11.2f}%"
    f"{lda_results[3] * 100:>11.2f}%"
)

print("=" * 70)
