import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Task 1: Generate Swiss roll dataset
n_samples = 1000
X, color = make_swiss_roll(n_samples=n_samples, noise=0.2, random_state=42)

# Task 2: Plot the resulting generated Swiss roll dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
plt.title("Generated Swiss Roll Dataset")
plt.show()

# Task 3: Use Kernel PCA with linear, RBF, and sigmoid kernels
# Apply Kernel PCA with linear kernel
kpca_linear = KernelPCA(kernel="linear")
X_kpca_linear = kpca_linear.fit_transform(X)

# Apply Kernel PCA with RBF kernel
kpca_rbf = KernelPCA(kernel="rbf", gamma=0.04)
X_kpca_rbf = kpca_rbf.fit_transform(X)

# Apply Kernel PCA with Polynomial kernel
kpca_poly = KernelPCA(kernel="poly", degree=3)  # You can adjust the degree as needed
X_kpca_poly = kpca_poly.fit_transform(X)

# Apply Kernel PCA with Polynomial kernel
kpca_poly = KernelPCA(kernel="poly", degree=3, gamma=1.0)  # Adjust gamma as needed
X_kpca_poly = kpca_poly.fit_transform(X)



# Task 4: Plot the kPCA results of applying the kernels and explain/compare
plt.figure(figsize=(12, 4))

# Subplot 1: kPCA with Linear Kernel
plt.subplot(131)
plt.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Task 4: kPCA with Linear Kernel")

# Subplot 2: kPCA with RBF Kernel
plt.subplot(132)
plt.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Task 4: kPCA with RBF Kernel")

# Subplot 3: kPCA with Polynomial Kernel
plt.subplot(133)
plt.scatter(X_kpca_poly[:, 0], X_kpca_poly[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Task 4: kPCA with Polynomial Kernel")

plt.tight_layout()
plt.show()

print("Task 4: The plots show the results of applying different kernels in kPCA.")


# Create binary class labels based on a threshold
threshold = 10  # Adjust the threshold as needed
y_binary = (color > threshold).astype(int)

# Task 5: Using kPCA and a kernel of your choice, apply Logistic Regression for classification.
# Use GridSearchCV to find the best kernel and gamma value for kPCA.
from sklearn.preprocessing import StandardScaler

# Define the parameter grid for GridSearchCV
param_grid = {
    "kpca__kernel": ["linear", "rbf", "poly"],
    "kpca__gamma": np.linspace(0.01, 0.1, 10),
}

# Create a pipeline including Kernel PCA, scaling, and Logistic Regression
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Scale the data before applying PCA
    ("kpca", KernelPCA()),
    ("logistic", LogisticRegression(max_iter=10000))
])

# Create a GridSearchCV object
grid_search = GridSearchCV(logistic_pipeline, param_grid, cv=3)

# Fit the GridSearchCV to your data with binary class labels
grid_search.fit(X, y_binary)  # Use y_binary as the target


# Task 6: Print out best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Task 6: Best Parameters:", best_params)


# Task 7: Plot the results from using GridSearchCV
results = grid_search.cv_results_
plt.figure(figsize=(10, 6))
plt.title("Task 7: GridSearchCV Results")
plt.xlabel("Gamma Value")
plt.ylabel("Mean Test Score")  # Update the ylabel to reflect the metric being plotted

# Store the kernel names in a list
kernels = ["linear", "rbf", "poly"]  # Include "poly" as it's one of the kernels used

# Create a colormap for different kernels
colormap = plt.cm.gist_ncar

for idx, kernel in enumerate(kernels):
    gamma_vals = np.linspace(0.01, 0.1, 10)
    mean_scores = results['mean_test_score'][results['param_kpca__kernel'] == kernel]
    
    # Plot each kernel with a unique color from the colormap
    plt.plot(gamma_vals, mean_scores, label=f"Kernel: {kernel}", color=colormap(idx / len(kernels)))

plt.legend()
plt.show()

