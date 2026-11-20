import pandas as pd

df = pd.read_csv("cleaned_phishing_data.csv", keep_default_na=False)

print(df.head())

print(df["Label"].value_counts())

df["Label"] = df["Label"].map({"phishing": 1, "legitimate": 0})

print(df.head())

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Cleaned_Text"], df["Label"], test_size=0.2, random_state=42
)

print(f"Training set size: {len(train_texts)}")
print(f"Testing set size: {len(test_texts)}")

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

import pickle

with open("processed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, train_labels, test_labels, vectorizer), f)

print("Processed data saved!")
