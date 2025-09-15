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
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

MAX_MEMORY = 3  # Keep last 3 messages for context
MAX_CHARS = 200  # Truncate each message to 200 chars

def get_context():
    """Combine last few messages as context for Gemini with truncation."""
    recent = st.session_state.chat_memory[-MAX_MEMORY:]
    context = ""
    for m in recent:
        user_text = m['user'][:MAX_CHARS]
        bot_text = m['bot'][:MAX_CHARS]
        context += f"User: {user_text}\nBot: {bot_text}\n"
    return context

def chat_with_gemini(prompt: str) -> str:
    """Generate response using Gemini with error handling."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Catch ResourceExhausted or other API errors
        return "⚠️ Sorry, I'm temporarily unable to respond. Please try again."

# -------------------------
# Display previous chat history
# -------------------------
for chat in st.session_state.chat_memory:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {chat['bot']}")

# -------------------------
# 💬 Input container
# -------------------------
user_input = st.chat_input("Type a message...")

if user_input:
    # Predict mood
    mood = predict_mood(user_input)

    # Decide how to call Gemini
    if mood in ["positive", "negative", "wrong_action"]:
        # Include context for mental wellness advice
        context = get_context()
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
        full_prompt = context + "\n\n" + prompt if context else prompt
        reply = chat_with_gemini(full_prompt)

        # Store in memory
        st.session_state.chat_memory.append({"user": user_input, "bot": reply, "mood": mood})

    else:
        # Treat as factual/general input — no chat context
        reply = chat_with_gemini(user_input)

    # Truncate memory if too long
    if len(st.session_state.chat_memory) > MAX_MEMORY:
        st.session_state.chat_memory = st.session_state.chat_memory[-MAX_MEMORY:]

    # Display current messages immediately
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {reply}")
