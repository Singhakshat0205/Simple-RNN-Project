import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# Load vocabulary and model
# ============================================================

@st.cache_resource
def load_resources():

    word_index = imdb.get_word_index()

    model = load_model("simple_rnn_imdb.h5")

    return word_index, model


word_index, model = load_resources()


# ============================================================
# Preprocess review
# ============================================================

MAX_FEATURES = 1000
MAX_LEN = 500

def preprocess_text(text):

    words = text.lower().split()

    encoded_review = []

    for word in words:

        index = word_index.get(word)

        # Unknown word OR word outside top 1000 vocabulary
        if index is None or index + 3 >= MAX_FEATURES:
            encoded_review.append(2)

        else:
            encoded_review.append(index + 3)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )

    return padded_review


# ============================================================
# Predict sentiment
# ============================================================

def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)

    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )

    probability = float(prediction[0][0])

    sentiment = (
        "Positive"
        if probability > 0.5
        else "Negative"
    )

    return sentiment, probability


# ============================================================
# Streamlit UI
# ============================================================

st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the Simple RNN will "
    "predict whether the sentiment is positive or negative."
)


# ============================================================
# User input
# ============================================================

review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was amazing and the acting was excellent!",
    height=150
)


# ============================================================
# Prediction
# ============================================================

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        sentiment, score = predict_sentiment(review)

        st.subheader("Prediction")

        # Display probability
        st.metric(
            "Positive Probability",
            f"{score * 100:.2f}%"
        )

        # Display sentiment
        if sentiment == "Positive":

            st.success(
                "😊 Positive Review"
            )

        else:

            st.error(
                "😞 Negative Review"
            )

        # Optional explanation
        st.write(
            f"Model prediction: **{sentiment}**"
        )
