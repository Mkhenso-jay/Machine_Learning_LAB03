# Exercise 9: Hyperparameter Tuning & Comparison
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# Standardize for classifiers that need it
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# ----------------------------
# Grid Search: SVM (example)
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10],
              'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_std, y_train)
print("Best SVM params:", grid.best_params_)
print("Best SVM CV score: %.2f" % grid.best_score_)

# ----------------------------
# Train other classifiers
models = {
    'Perceptron': Perceptron(max_iter=40, eta0=0.1, random_state=1),
    'LogisticRegression': LogisticRegression(C=100.0, random_state=1, max_iter=1000, multi_class='ovr'),
    'SVM Linear': SVC(kernel='linear', C=1.0, random_state=1),
    'DecisionTree': DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1),
    'RandomForest': RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2),
    'KNN': KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
}

# Train and evaluate
accuracy_results = {}
for name, clf in models.items():
    if name in ['Perceptron', 'LogisticRegression', 'SVM Linear', 'KNN']:
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    accuracy_results[name] = accuracy_score(y_test, y_pred)

# Print comparison table
print("\nClassifier Accuracy Comparison:")
for name, acc in accuracy_results.items():
    print(f"{name}: {acc:.2f}")
