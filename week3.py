import pickle

with open("processed_data.pkl", "rb") as f:
    X_train, X_test, train_labels, test_labels, vectorizer = pickle.load(f)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, train_labels)

print("Model training complete!")

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Trained model saved!")
