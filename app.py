# app.py

import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

max_features = 10000   
max_len = 500          

# IMDB word index (word -> integer)
word_index = imdb.get_word_index()
# reverse index - is useful for decoding
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained LSTM model from your NLP folder
model = load_model("imdb_lstm_fjoertoft.h5")
)

#Helper functions

def decode_review(encoded_review):
    """Convert list of integers back to text (for examples)."""
    return " ".join(
        [reverse_word_index.get(i - 3, "?") for i in encoded_review]
    )

def preprocess_text(text: str):
    """Turn raw text from user into a padded sequence, same as training."""
    words = text.lower().split()
    encoded_review = []

    for word in words:
        idx = word_index.get(word)  # original index from IMDB word_index

        if idx is not None and idx < max_features:
            # valid word within our 10,000-word vocab → apply +3 offset
            encoded_idx = idx + 3
        else:
            # unknown or too-rare word → use <UNK> token (2)
            encoded_idx = 2

        encoded_review.append(encoded_idx)

    padded_review = sequence.pad_sequences(
        [encoded_review], maxlen=max_len
    )
    return padded_review

def predict_sentiment(review_text: str):
    """Return sentiment label and probability."""
    preprocessed_input = preprocess_text(review_text)
    prediction = model.predict(preprocessed_input, verbose=0)
    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score

# Step 3: Streamlit UI

st.title("IMDB Movie Review Classifier by Fjoertoft")
st.write(
    "Enter a movie review below to classify it as positive or negative. "
    "This app uses an LSTM model trained on the IMDB dataset."
)

user_input = st.text_area("Movie review:")

if st.button("Classify review"):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction score:** {score:.4f}")
    else:
        st.warning("Please enter a review first.")

st.markdown("---")

# 5 example: reviews and predictions
st.subheader("Example IMDB reviews and model predictions")

# Load some test reviews for examples 
@st.cache_data
def load_example_reviews(n=5):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    X_test_padded = sequence.pad_sequences(X_test, maxlen=max_len)
    # take the first n for simplicity
    return X_test[:n], X_test_padded[:n], y_test[:n]

encoded_examples, padded_examples, true_labels = load_example_reviews(5)

for i in range(len(encoded_examples)):
    raw_text = decode_review(encoded_examples[i])
    pred = model.predict(padded_examples[i:i+1], verbose=0)
    score = float(pred[0][0])
    pred_label = "Positive" if score > 0.5 else "Negative"
    true_label = "Positive" if true_labels[i] == 1 else "Negative"

    st.write(f"**Example {i+1}:**")
    st.write(raw_text)
    st.write(f"*True label:* {true_label}")
    st.write(f"*Predicted:* {pred_label}  (score = {score:.4f})")
    st.write("---")
