# Exercise 1: Data Preparation
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal length and petal width
y = iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# Standardize features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Combine train/test for plotting later
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

print("Class distribution (training set):", np.bincount(y_train))
print("Training set shape:", X_train_std.shape)
print("Test set shape:", X_test_std.shape)

# Nonlinear dataset (moons)
X_moons, y_moons = make_moons(n_samples=100, noise=0.2, random_state=123)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=1)

sc_m = StandardScaler()
X_train_m_std = sc_m.fit_transform(X_train_m)
X_test_m_std = sc_m.transform(X_test_m)

print("Moons dataset prepared!")
