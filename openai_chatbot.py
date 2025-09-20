import streamlit as st
import google.generativeai as genai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from gtts import gTTS
import tempfile
import speech_recognition as sr
from streamlit_audiorecorder import audiorecorder  # updated package

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
# 🎤 Voice Output Helper
# -------------------------
def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        st.audio(tmp.name, format="audio/mp3")

# -------------------------
# 🌐 Streamlit UI
# -------------------------
st.set_page_config(page_title="Youth Mental Wellness Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Youth Mental Wellness Chatbot (Voice + Text)")

# Initialize chat memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

MAX_MEMORY = 3
MAX_CHARS = 200

def get_context():
    recent = st.session_state.chat_memory[-MAX_MEMORY:]
    context = ""
    for m in recent:
        user_text = m['user'][:MAX_CHARS]
        bot_text = m['bot'][:MAX_CHARS]
        context += f"User: {user_text}\nBot: {bot_text}\n"
    return context

def chat_with_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return "⚠️ Sorry, I'm temporarily unable to respond. Please try again."

# -------------------------
# Display previous chat
# -------------------------
for chat in st.session_state.chat_memory:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {chat['bot']}")

# -------------------------
# 💬 Input (Text + Voice)
# -------------------------
col1, col2 = st.columns([2,1])

with col1:
    user_input = st.chat_input("Type a message...")

with col2:
    st.write("🎤 Or record your voice:")
    audio_bytes = audiorecorder("🎙️ Start Recording", "⏹ Stop Recording")

    if audio_bytes:
        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_filename = tmp_file.name

        # Convert voice to text
        r = sr.Recognizer()
        with sr.AudioFile(tmp_filename) as source:
            audio_data = r.record(source)
            try:
                user_input = r.recognize_google(audio_data)
                st.success(f"🗣️ You said: {user_input}")
            except sr.UnknownValueError:
                st.error("❌ Sorry, I couldn't understand that.")
                user_input = None
            except sr.RequestError:
                st.error("⚠️ Speech recognition service unavailable.")
                user_input = None

# -------------------------
# Process Input
# -------------------------
if user_input:
    mood = predict_mood(user_input)

    if mood in ["positive", "negative", "wrong_action"]:
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
        st.session_state.chat_memory.append({"user": user_input, "bot": reply, "mood": mood})
    else:
        reply = chat_with_gemini(user_input)

    # Truncate memory
    if len(st.session_state.chat_memory) > MAX_MEMORY:
        st.session_state.chat_memory = st.session_state.chat_memory[-MAX_MEMORY:]

    # Display chat
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {reply}")

    # 🔊 Speak reply
    speak_text(reply)

