import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_digits
from pca_from_scratch import PCAFromScratch

digits = load_digits()
X = digits.data

pca = PCAFromScratch(64)
pca.fit(X)

variance = pca.explained_variance_ratio_
cumulative = np.cumsum(variance)

print("=" * 60)
print("PCA VISUALIZATION")
print("=" * 60)

print("Original dimensions:", X.shape[1])
print("80% variance:", np.argmax(cumulative >= 0.80) + 1)
print("90% variance:", np.argmax(cumulative >= 0.90) + 1)
print("95% variance:", np.argmax(cumulative >= 0.95) + 1)
print("20 components:", cumulative[19] * 100, "%")

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 65), variance * 100, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("Explained Variance of Principal Components")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/pca_explained_variance.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 65), cumulative * 100, marker="o")
plt.axhline(90, linestyle="--", label="90% variance")
plt.axvline(20, linestyle="--", label="20 components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/pca_cumulative_variance.png", dpi=300)
plt.close()

print()
print("Graphs saved successfully.")
print("results/pca_explained_variance.png")
print("results/pca_cumulative_variance.png")
