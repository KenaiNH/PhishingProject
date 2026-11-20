import pickle

with open("best_phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("processed_data.pkl", "rb") as f:
    _, _, _, _, vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully!")

import numpy as np

def classify_email(email_text):

    email_vectorized = vectorizer.transform([email_text])

    prediction = model.predict(email_vectorized)

    result = "Phishing Email 🚨" if prediction[0] == 1 else "Legitimate Email ✅"
    return result

test_email = "Click this link to reset your password: http://fakebank.com"
print("Test Email Classification:", classify_email(test_email))

if __name__ == "__main__":
    print("🔍 Phishing Detection Tool 🔍")
    print("Type or paste an email message below and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter email text: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print("Result:", classify_email(user_input))
        print("\n")

with open("final_phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Final model saved for future use!")
