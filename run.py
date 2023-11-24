from sklearn.feature_extraction.text import CountVectorizer
import joblib
from faker import Faker
import numpy as np

# Load the saved model
saved_model = joblib.load("clf_model.h5")

# Load the vectorizer trained during training
count_vectorizer = joblib.load("count_vectorizer.h5")

faker_instance = Faker()

# Input data
inp = np.array([faker_instance.sentence() for _ in range(50)])

# Transform the input data using the same vectorizer
vectorized_inp = count_vectorizer.transform(inp)

# Make predictions using the saved model
predictions = saved_model.predict(vectorized_inp)

# Find the maximum length of an email to determine the column width
max_len = max(len(email) for email in inp)

# Print emails, predictions, and translations in a symmetric way
for email, prediction in zip(inp, predictions):
    print(f"Email: {email.ljust(max_len)} | Prediction: {prediction}\n")
