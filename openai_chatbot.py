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

# -------------------------
# 🌐 Streamlit UI
# -------------------------
st.set_page_config(page_title="Youth Mental Wellness Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Youth Mental Wellness Chatbot")

# Initialize session state for chat memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

MAX_MEMORY = 10  # store only last 5 mood-related messages

def get_context():
    """Combine last few messages as context for Gemini."""
    return "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in st.session_state.chat_memory])

def chat_with_gemini(prompt: str) -> str:
    """Generate response using Gemini."""
    response = gemini_model.generate_content(prompt)
    return response.text

# Display chat history first
for chat in st.session_state.chat_memory:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {chat['bot']}")

# -------------------------
# 💬 Input container moved to the bottom
# -------------------------
user_input = st.chat_input("💬 Type a message...")

if user_input:
    # Predict mood
    mood = predict_mood(user_input)

    # Decide how to call Gemini
    if mood in ["positive", "negative", "wrong_action"]:
        # Include context for mental wellness advice
        context = get_context()
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
        full_prompt = context + "\n\n" + prompt if context else prompt
        reply = chat_with_gemini(full_prompt)

        # Store in memory (only mood-related)
        st.session_state.chat_memory.append({"user": user_input, "bot": reply, "mood": mood})
        if len(st.session_state.chat_memory) > MAX_MEMORY:
            st.session_state.chat_memory.pop(0)

    else:
        # Treat as factual/general input — no context
        reply = chat_with_gemini(user_input)
    
