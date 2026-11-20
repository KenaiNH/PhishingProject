import pickle

with open("processed_data.pkl", "rb") as f:
    X_train, X_test, train_labels, test_labels, vectorizer = pickle.load(f)

with open("phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model and data loaded successfully!")

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "max_iter": [500, 1000, 2000]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="accuracy", verbose=1)

grid_search.fit(X_train, train_labels)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

from sklearn.svm import SVC

svm_model = SVC(kernel="linear", C=1)
svm_model.fit(X_train, train_labels)

svm_predictions = svm_model.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, train_labels)

rf_predictions = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score

lr_accuracy = accuracy_score(test_labels, best_model.predict(X_test))
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")

svm_accuracy = accuracy_score(test_labels, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

rf_accuracy = accuracy_score(test_labels, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

import numpy as np

class_counts = np.bincount(train_labels)
print("Phishing emails:", class_counts[1])
print("Legitimate emails:", class_counts[0])

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_balanced, train_labels_balanced = smote.fit_resample(X_train, train_labels)

print("After SMOTE, training set size:", X_train_balanced.shape)

best_final_model = svm_model

with open("best_phishing_model.pkl", "wb") as f:
    pickle.dump(best_final_model, f)

print("Best model saved!")
