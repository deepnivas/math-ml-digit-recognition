import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_digits

digits = load_digits()
image = digits.images[0]

print("=" * 60)
print("SVD RECONSTRUCTION")
print("=" * 60)
print("Original image shape:", image.shape)

U, S, VT = np.linalg.svd(image, full_matrices=False)

print("Number of singular values:", len(S))

print("\nSingular values:")
for i in range(len(S)):
    print(f"{i + 1}: {S[i]:.4f}")

ranks = [1, 2, 3, 4, 5, 8]
reconstructed_images = []

for rank in ranks:
    reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ VT[:rank, :]
    reconstructed_images.append(reconstructed)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(14, 7))

plt.subplot(2, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

for i in range(len(ranks)):
    plt.subplot(2, 4, i + 2)
    plt.imshow(reconstructed_images[i], cmap="gray")
    plt.title(f"Rank {ranks[i]}")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/svd_reconstruction_comparison.png",
    dpi=300
)

plt.close()

total_energy = np.sum(S ** 2)

print("\nInformation retained:")

for rank in ranks:
    retained = np.sum(S[:rank] ** 2) / total_energy
    print(f"Rank {rank}: {retained * 100:.2f}%")

print("\nSVD visualization saved:")
print("results/svd_reconstruction_comparison.png")
