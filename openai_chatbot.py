import streamlit as st
import google.generativeai as genai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------
# 🌟 Configure Gemini API
# -------------------------
genai.configure(api_key=st.secrets["gemini_api_key"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------
# 🧠 Simple ML Mood/Action Model
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

def chat_with_gemini(prompt: str) -> str:
    response = gemini_model.generate_content(prompt)
    return response.text

# -------------------------
# 🌐 Streamlit UI
# -------------------------
st.set_page_config(page_title="Youth Mental Wellness Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Youth Mental Wellness Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input container
with st.container():
    user_input = st.text_input("💬 Type a message...", key="input_text")
    send_button = st.button("Send")

if send_button and user_input.strip() != "":
    # ML prediction
    mood = predict_mood(user_input)

    # Generate Gemini response
    if mood == "wrong_action":
        prompt = (
            f"You are a cool, honest mental wellness AI. The user admitted a harmful action: {user_input}. "
            "Do NOT validate it. Explain why it was wrong and suggest practical ways to improve. "
            "Be calm, supportive, and actionable."
        )
    else:
        prompt = (
            f"You are a cool, improvement-focused mental wellness AI. The user feels {mood}: {user_input}. "
            "Do NOT just sympathize; provide practical advice or perspective to improve their mental wellness."
        )
    reply = chat_with_gemini(prompt)

    # Append messages to session state
    st.session_state.messages.append({"user": user_input, "bot": reply, "mood": mood})

# Display chat history dynamically
for chat in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {chat['bot']}")
