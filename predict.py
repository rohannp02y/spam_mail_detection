import pickle

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Test with different emails
test_emails = [
    "I have a date with pookie on Monday.",
    "Click here to win free money now!!!",
    "Meeting scheduled for tomorrow at 2 PM",
    "Congratulations! You've won a lottery prize!"
]

print("\n=== SPAM DETECTION RESULTS ===\n")

for email in test_emails:
    # Convert email to features
    email_features = vectorizer.transform([email])
    
    # Make prediction
    prediction = model.predict(email_features)
    
    # Show result
    if prediction[0] == 0:
        result = "SPAM"
    else:
        result = "HAM (Legitimate)"
    
    print(f"Email: {email}")
    print(f"Result: {result}")
    print()
