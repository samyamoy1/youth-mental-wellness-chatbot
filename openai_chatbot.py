import streamlit as st
import google.generativeai as genai
from transformers import pipeline

# -------------------------
# ⚙️ Configure Gemini API
# -------------------------
genai.configure(api_key=st.secrets["gemini_api_key"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------
# 🧠 Sentiment Analyzer (HuggingFace)
# -------------------------
sentiment_analyzer = pipeline("sentiment-analysis")

def predict_mood(user_input: str) -> str:
    result = sentiment_analyzer(user_input)[0]["label"]
    return result.lower()   # "positive", "negative", "neutral"

# -------------------------
# 🤖 Gemini Response
# -------------------------
def chat_with_gemini(user_input: str, mood: str, history: list) -> str:
    # Convert history into text
    history_text = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:]])

    # Mood-based style guide
    if mood == "positive":
        style = "Celebrate their happiness, use cheerful energy and emojis like 😊🌟."
    elif mood == "negative":
        style = "Be extra gentle, empathetic, and comforting 💙. Acknowledge their struggle."
    else:
        style = "Respond in a warm, neutral, caring tone."

    prompt = f"""
You are a supportive friend for youth mental wellness.
- Speak casually, like a kind and empathetic buddy.
- Always first acknowledge how they feel.
- Keep sentences short and natural, not robotic.
- {style}

Conversation so far:
{history_text}

Now, reply to the user's new message.

User: {user_input}
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# -------------------------
# 🌐 Streamlit UI
# -------------------------
st.set_page_config(page_title="Youth Mental Wellness Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Youth Mental Wellness Chatbot")
st.write("An AI-powered, confidential, and empathetic chatbot for youth mental health support.")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_area("💬 How are you feeling today?", "")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Step 1: Mood detection
        mood = predict_mood(user_input)

        # Step 2: Get Gemini response
        reply = chat_with_gemini(user_input, mood, st.session_state["history"])

        # Save to history
        st.session_state["history"].append({"user": user_input, "bot": reply})

# -------------------------
# 📜 Show Conversation
# -------------------------
if st.session_state["history"]:
    st.subheader("🤗 Chat with your Wellness Buddy")
    for chat in st.session_state["history"]:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

st.markdown("---")
st.caption("⚠️ Disclaimer: This chatbot is not a substitute for professional help. If you are struggling, please reach out to a licensed mental health professional.")

