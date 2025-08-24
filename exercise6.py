# Exercise 6: Random Forests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# Helper function: decision region plot
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black',
                    alpha=1.0, linewidth=1, marker='o', s=100, label='test set')
    plt.legend(loc='upper left')
    plt.show()


# Load Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# Train Random Forest
forest = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

# Predictions
y_pred = forest.predict(X_test)
print("Random Forest Accuracy: %.2f" % accuracy_score(y_test, y_pred))

# Combine train/test for plotting
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
test_idx = range(len(y_train), len(y_train) + len(y_test))

# Plot decision regions
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=test_idx)

# Optional: feature importances
print("Feature importances:", forest.feature_importances_)
