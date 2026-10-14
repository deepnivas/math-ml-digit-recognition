import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from pca_from_scratch import PCAFromScratch
from lda_from_scratch import LDAFromScratch


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


# PCA: 64 -> 20 dimensions
pca = PCAFromScratch(n_components=20)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# LDA: 20 -> 9 dimensions
lda = LDAFromScratch(n_components=9)

X_train_lda = lda.fit_transform(
    X_train_pca,
    y_train
)

X_test_lda = lda.transform(
    X_test_pca
)


# KNN classifier
model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train_lda,
    y_train
)


def predict_digit(image):

    image = np.asarray(image).reshape(1, -1)

    image_pca = pca.transform(image)

    image_lda = lda.transform(image_pca)

    prediction = model.predict(image_lda)[0]

    probabilities = model.predict_proba(image_lda)[0]

    confidence = probabilities[prediction]

    return prediction, confidence


if __name__ == "__main__":

    sample = X_test[0]

    actual_digit = y_test[0]

    prediction, confidence = predict_digit(sample)

    print("=" * 60)
    print("DIGIT PREDICTION")
    print("=" * 60)

    print("Actual digit    :", actual_digit)
    print("Predicted digit :", prediction)
    print(
        "Confidence      :",
        f"{confidence * 100:.2f}%"
    )

    print("=" * 60)
