# fulllab03.py
# Full Lab 03: Machine Learning Classifiers
# Author: Your Name
# Python Machine Learning, Chapter 3 exercises
# --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
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

    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black',
                    alpha=1.0, linewidth=1, marker='o', s=100, label='test set')
    plt.legend(loc='upper left')
    plt.show()

# ----------------------------
# Exercise 1: Data Preparation
print("----- Exercise 1: Data Preparation -----")
iris = load_iris()
X = iris.data[:, [2, 3]]  # petal length and width
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print("Class distribution:", np.bincount(y_train))

# ----------------------------
# Exercise 2: Perceptron
print("\n----- Exercise 2: Perceptron -----")
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print("Perceptron Accuracy:", accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined_std, y_combined, classifier=ppn, test_idx=range(len(y_train), len(y_combined)))

# ----------------------------
# Exercise 3: Logistic Regression
print("\n----- Exercise 3: Logistic Regression -----")
lr = LogisticRegression(C=100.0, random_state=1, max_iter=1000, multi_class='ovr')
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(len(y_train), len(y_combined)))

# ----------------------------
# Exercise 4: Support Vector Machines (SVM)
print("\n----- Exercise 4: SVM -----")
svm_linear = SVC(kernel='linear', C=1.0, random_state=1)
svm_linear.fit(X_train_std, y_train)
y_pred = svm_linear.predict(X_test_std)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined_std, y_combined, classifier=svm_linear, test_idx=range(len(y_train), len(y_combined)))

# ----------------------------
# Exercise 5: Decision Trees
print("\n----- Exercise 5: Decision Tree -----")
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
X_combined = np.vstack((X_train, X_test))
y_combined_tree = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined_tree, classifier=tree, test_idx=range(len(y_train), len(y_combined_tree)))

# ----------------------------
# Exercise 6: Random Forests
print("\n----- Exercise 6: Random Forest -----")
forest = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined, y_combined_tree, classifier=forest, test_idx=range(len(y_train), len(y_combined_tree)))
print("Random Forest Feature Importances:", forest.feature_importances_)

# ----------------------------
# Exercise 7: K-Nearest Neighbors (KNN)
print("\n----- Exercise 7: KNN -----")
knn = KNeighborsClassifier(n_neighbors=5, p=2)
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(len(y_train), len(y_combined)))

# ----------------------------
# Exercise 8: Kernel SVM (Moons Dataset)
print("\n----- Exercise 8: Kernel SVM (Moons) -----")
X_moons, y_moons = make_moons(n_samples=100, noise=0.2, random_state=123)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_moons, y_moons, test_size=0.3, random_state=1)
sc_m = StandardScaler()
X_train_m_std = sc_m.fit_transform(X_train_m)
X_test_m_std = sc_m.transform(X_test_m)
X_combined_m_std = np.vstack((X_train_m_std, X_test_m_std))
y_combined_m = np.hstack((y_train_m, y_test_m))
svm_rbf = SVC(kernel='rbf', gamma=0.2, C=1.0, random_state=1)
svm_rbf.fit(X_train_m_std, y_train_m)
y_pred_m = svm_rbf.predict(X_test_m_std)
print("Kernel SVM (Moons) Accuracy:", accuracy_score(y_test_m, y_pred_m))
plot_decision_regions(X_combined_m_std, y_combined_m, classifier=svm_rbf, test_idx=range(len(y_train_m), len(y_combined_m)))

# ----------------------------
# Exercise 9: Hyperparameter Tuning & Comparison
print("\n----- Exercise 9: Hyperparameter Tuning & Comparison -----")
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_std, y_train)
print("Best SVM params:", grid.best_params_)
print("Best SVM CV score: %.2f" % grid.best_score_)

# Train all classifiers for comparison
models = {
    'Perceptron': ppn,
    'LogisticRegression': lr,
    'SVM Linear': svm_linear,
    'DecisionTree': tree,
    'RandomForest': forest,
    'KNN': knn
}

print("\nClassifier Accuracy Comparison:")
for name, clf in models.items():
    if name in ['Perceptron', 'LogisticRegression', 'SVM Linear', 'KNN']:
        y_pred = clf.predict(X_test_std)
    else:
        y_pred = clf.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, y_pred):.2f}")
