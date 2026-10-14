import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def reconstruct_image(U, S, VT, k):
    return U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]


# Load handwritten digit dataset
digits = load_digits()
images = digits.images

print("=" * 60)
print("SVD AND LOW-RANK APPROXIMATION")
print("=" * 60)

print(f"Number of images : {len(images)}")
print(f"Image dimensions : {images.shape[1]} x {images.shape[2]}")


# Select one handwritten digit
image = images[0]

print("\nOriginal image matrix:")
print(image)


# Perform SVD
U, S, VT = np.linalg.svd(
    image,
    full_matrices=False
)

print("\nSVD dimensions:")
print(f"U  : {U.shape}")
print(f"S  : {S.shape}")
print(f"VT : {VT.shape}")

print("\nSingular values:")

for i, value in enumerate(S, start=1):
    print(f"sigma{i} = {value:.4f}")


# Calculate information retained
energy = S ** 2
explained_ratio = energy / np.sum(energy)
cumulative_ratio = np.cumsum(explained_ratio)

print("\nInformation retained:")
print("-" * 40)

for k in range(1, len(S) + 1):
    print(
        f"k = {k}: "
        f"{cumulative_ratio[k - 1] * 100:.2f}%"
    )


# Low-rank reconstruction
ranks = [1, 2, 3, 4, 5, 6, 7, 8]

fig, axes = plt.subplots(
    2,
    4,
    figsize=(10, 5)
)

for ax, k in zip(axes.flat, ranks):

    reconstructed = reconstruct_image(
        U,
        S,
        VT,
        k
    )

    ax.imshow(
        reconstructed,
        cmap="gray"
    )

    ax.set_title(
        f"k = {k}\n"
        f"{cumulative_ratio[k - 1] * 100:.1f}% retained"
    )

    ax.axis("off")


plt.suptitle(
    "SVD Low-Rank Reconstruction of a Handwritten Digit",
    fontsize=14
)

plt.tight_layout()

output_path = "results/svd_reconstruction.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print("\n" + "=" * 60)
print(f"Visualization saved to: {output_path}")
print("=" * 60)

plt.show()
