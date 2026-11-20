import pandas as pd

df = pd.read_csv("phishing.csv")

print(df.head())

print(df.isnull().sum())

df = df.dropna()

import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    return text

df["Cleaned_Text"] = df["Email Text"].apply(clean_text)

print(df[["Email Text", "Cleaned_Text"]].head())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sample_text = df["Cleaned_Text"][0]
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original Text:", sample_text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

df[["Cleaned_Text", "Label"]].to_csv("cleaned_phishing_data.csv", index=False)
print("Cleaned data saved!")
