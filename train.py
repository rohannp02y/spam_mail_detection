import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the data
raw_mail_data = pd.read_csv("mail_data.csv")
print("Data loaded!")

# Replace empty values
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# Convert spam/ham to 0/1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate message and category
X = mail_data["Message"]
Y = mail_data["Category"]

# Split into train (80%) and test (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
feature_extraction = TfidfVectorizer(min_df = 1, stop_words= 'english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Check accuracy
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print("Accuracy on training data: ", accuracy_on_training_data)
print("Accuracy on test data: ", accuracy_on_test_data)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(feature_extraction, open('vectorizer.pkl', 'wb'))

print("Model saved!")