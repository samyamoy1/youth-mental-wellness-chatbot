import streamlit as st
import google.generativeai as genai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# 🔑 Configure Gemini API using Streamlit Secrets
# -------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------
# 🧠 Simple ML Sentiment Model (toy training)
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
        "I am confident and strong"
    ],
    "label": [
        "positive", "positive",
        "negative", "negative",
        "negative", "positive",
        "negative", "positive"
    ]
}

df = pd.DataFrame(training_data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

ml_model = LogisticRegression()
ml_model.fit(X, y)

def predict_mood(user_input: str) -> str:
    X_test = vectorizer.transform([user_input])
    return ml_model.predict(X_test)[0]

# -------------------------
# 🤖 Gemini Chat Function
# -------------------------
def chat_with_gemini(prompt: str) -> str:
    response = gemini_model.generate_content(prompt)
    return response.text

# -------------------------
# 🌐 Streamlit UI
# -------------------------
st.set_page_config(page_title="Youth Mental Wellness Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Youth Mental Wellness Chatbot")
st.write("An AI-powered, confidential, and empathetic chatbot for youth mental health support.")

user_input = st.text_area("💬 How are you feeling today?", "")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Step 1: ML prediction
        mood = predict_mood(user_input)

        # Step 2: Generate Gemini response
        prompt = f"You are a kind and empathetic mental health assistant. The user feels {mood}. Respond in a caring way: {user_input}"
        reply = chat_with_gemini(prompt)

        st.subheader("🤗 Chatbot Response:")
        st.write(reply)

        st.subheader("📝 Mood Analysis (ML Model):")
        st.write(f"Predicted Mood: **{mood.capitalize()}**")

st.markdown("---")
st.caption("⚠️ Disclaimer: This chatbot is not a substitute for professional help. If you are struggling, please reach out to a licensed mental health professional.")
