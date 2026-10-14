import matplotlib.pyplot as plt
import os

methods = [
    "Raw Pixels",
    "PCA",
    "PCA + LDA"
]

accuracy = [
    98.33,
    98.61,
    97.22
]

dimensions = [
    64,
    20,
    9
]

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(9, 6))

bars = plt.bar(
    methods,
    accuracy
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Method")
plt.title("Classification Accuracy Comparison")

plt.ylim(90, 100)

for bar, value, dim in zip(bars, accuracy, dimensions):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.1,
        f"{value}%\n({dim} features)",
        ha="center"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/model_accuracy_comparison.png",
    dpi=300
)

plt.close()

print("Comparison graph saved:")
print("results/model_accuracy_comparison.png")
