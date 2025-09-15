import streamlit as st
import google.generativeai as genai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------
# Configure Gemini API
# -------------------------
genai.configure(api_key=st.secrets["gemini_api_key"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------
# Simple ML Mood/Action Model
# -------------------------
training_data = {
    "text": [
        "I am feeling very happy today",
        "Life is beautiful and I am excited",
        "I am so sad and depressed",
        "I feel anxious about exams",
        "I am stressed and worried",
        "Everything is going well",
        "I feel alone and hopeless",
        "I am confident and strong",
        "I did something bad today",
        "I hurt someone"
    ],
    "label": [
        "positive", "positive",
        "negative", "negative",
        "negative", "positive",
        "negative", "positive",
        "wrong_action", "wrong_action"
    ]
}

df = pd.DataFrame(training_data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

ml_model = MultinomialNB()
ml_model.fit(X, y)

def predict_mood(user_input: str) -> str:
    X_test = vectorizer.transform([user_input])
    return ml_model.predict(X_test)[0]

# -------------------------
# Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="Youth Mental Wellness Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Youth Mental Wellness Chatbot")

def chat_with_gemini(prompt: str) -> str:
    """Generate response using Gemini with error handling."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "⚠️ Sorry, I'm temporarily unable to respond. Please try again."

# -------------------------
# Input container
# -------------------------
user_input = st.chat_input("💬 Type a message...")

if user_input:
    mood = predict_mood(user_input)

    if mood in ["positive", "negative", "wrong_action"]:
        if mood == "wrong_action":
            prompt = (
                f"You are a calm, supportive mental wellness AI. "
                f"The user admitted a harmful action: {user_input}. "
                "Do NOT validate it. Explain why it was wrong and suggest practical ways to improve."
            )
        else:
            prompt = (
                f"You are a calm, improvement-focused mental wellness AI. "
                f"The user feels {mood}: {user_input}. "
                "Do NOT just sympathize; provide practical advice or perspective to improve mental wellness."
            )
    else:
        prompt = user_input  # No special context for general/factual input

    reply = chat_with_gemini(prompt)

    # Display interaction immediately (no memory stored)
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write(reply)
