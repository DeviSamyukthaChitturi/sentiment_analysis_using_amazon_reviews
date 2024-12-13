import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Load the saved model and tokenizer
model = load_model('/Users/chitturidevisamyuktha/Desktop/streamlit_sentiment_app/sentiment_lstm_model.h5')

# Load the tokenizer (Ensure it's saved as tokenizer.pkl in the same directory as app.py)
with open('/Users/chitturidevisamyuktha/Desktop/streamlit_sentiment_app/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Streamlit UI
st.title("Sentiment Analysis with LSTM")

# Display background image for the app
image = Image.open('/Users/chitturidevisamyuktha/Desktop/streamlit_sentiment_app/BG_Image.jpg')  # Add the path to your image file here
st.image(image, use_column_width=True)

# Text input
input_text = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    if input_text.strip():
        # Preprocess the input
        sequences = tokenizer.texts_to_sequences([input_text])
        padded_sequences = pad_sequences(sequences, maxlen=200)

        # Predict the sentiment
        prediction = model.predict(padded_sequences)
        sentiment = "Positive" if prediction > 0.5 else "Negative"

        # Display the result
        st.success(f"Predicted Sentiment: *{sentiment}*")
    else:
        st.warning("Please enter text for prediction.")