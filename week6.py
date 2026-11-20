import pickle

with open("final_phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("processed_data.pkl", "rb") as f:
    _, _, _, _, vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully!")

test_emails = [
    "Your account has been suspended! Click this link to restore access: http://fake-bank.com",
    "Meeting tomorrow at 10 AM. Let me know if you can make it.",
    "URGENT: Your PayPal account was compromised! Log in now: http://phishingsite.com",
    "Hey, let's grab lunch today at noon!",
    "Dear customer, confirm your identity by clicking this link: http://secure-verification.com",
]

def classify_email(email_text):
    email_vectorized = vectorizer.transform([email_text])
    prediction = model.predict(email_vectorized)
    return "Phishing Email 🚨" if prediction[0] == 1 else "Legitimate Email ✅"

for i, email in enumerate(test_emails):
    print(f"Email {i+1}: {classify_email(email)}")

actual_labels = [1, 0, 1, 0, 1]
predicted_labels = [model.predict(vectorizer.transform([email]))[0] for email in test_emails]

for i in range(len(test_emails)):
    if actual_labels[i] != predicted_labels[i]:
        print(f"❌ Incorrect Prediction for Email {i+1}:")
        print("   Email Text:", test_emails[i])
        print("   Actual Label:", "Phishing" if actual_labels[i] == 1 else "Legitimate")
        print("   Predicted:", "Phishing" if predicted_labels[i] == 1 else "Legitimate")
        print()
