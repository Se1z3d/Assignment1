import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA

# Task 1: Load the MNIST dataset with 70,000 instances
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target

# Task 2: Display each digit
def display_digits(X, y, num_digits=10):
    fig, axes = plt.subplots(1, num_digits, figsize=(15, 3))
    for i in range(num_digits):
        digit_image = X.iloc[i].values.reshape(28, 28)  # Use iloc to access DataFrame rows and .values to get a numpy array
        axes[i].imshow(digit_image, cmap=plt.cm.binary, interpolation="nearest")
        axes[i].set_title(f"Label: {y.iloc[i]}")  # Use iloc for accessing labels as well
        axes[i].axis("off")
    plt.show()

display_digits(X, y)


# Task 3: Use PCA to retrieve the 1st and 2nd principal components and output their explained variance ratio
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio for 1st and 2nd principal component: {explained_variance_ratio}")

# Task 4: Plot the projections of the 1st and 2nd principal components onto a 1D hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(label='Digit Label')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('PCA Projection of MNIST Data')
plt.show()

# Task 5: Use Incremental PCA to reduce dimensionality to 154 dimensions
n_components = 154
ipca = IncrementalPCA(n_components=n_components)
X_ipca = ipca.fit_transform(X)

# Task 6: Display the original and compressed digits
def display_compressed_vs_original(X_original, X_compressed, num_digits=10):
    fig, axes = plt.subplots(2, num_digits, figsize=(15, 6))
    for i in range(num_digits):
        original_image = X_original.iloc[i].values.reshape(28, 28)
        compressed_image = X_compressed[i].reshape(28, 28)
        
        axes[0, i].imshow(original_image, cmap=plt.cm.binary, interpolation="nearest")
        axes[0, i].set_title(f"Original Label: {y.iloc[i]}")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(compressed_image, cmap=plt.cm.binary, interpolation="nearest")
        axes[1, i].set_title("Compressed")
        axes[1, i].axis("off")
    plt.show()

display_compressed_vs_original(X, ipca.inverse_transform(X_ipca))

