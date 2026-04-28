import streamlit as st
import google.generativeai as genai
import PyPDF2
import os
import glob

# --- 1. SETUP GEMINI AI ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. BUILD THE WEBPAGE ---
st.set_page_config(page_title="Huliot AI Assistant", page_icon="💧")
st.title("💧 Huliot Technical Assistant")
st.write("Ask me anything! Or type **LEARN:** to teach me a new rule!")

# --- 3. READ ALL PDF FILES ---
@st.cache_data
def load_knowledge_base():
    text = ""
    pdf_files = glob.glob("*.pdf") 
    for file_name in pdf_files:
        try:
            with open(file_name, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception:
            pass
    return text, len(pdf_files)

catalog_text, file_count = load_knowledge_base()

# --- 4. READ THE DIARY ---
diary_memory = ""
if os.path.exists("robot_diary.txt"):
    with open("robot_diary.txt", "r", encoding="utf-8") as file:
        diary_memory = file.read()

with st.sidebar:
    st.header("🧠 AI Brain Status")
    st.success(f"{file_count} PDFs Memorized!")
    if diary_memory != "":
        st.info("📓 Diary has saved memories!")
    else:
        st.warning("📓 Diary is empty right now.")
        
    st.divider()
    st.header("⚙️ AI Settings")
    st.write("Select an AI Brain that has free credits left:")
    
    # THE FIX: Bringing back your dropdown menu!
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model = st.selectbox("Choose AI Brain", available_models)
    except Exception as e:
        st.error("Could not fetch models. Check API Key.")
        selected_model = "models/gemini-1.5-flash"

# --- 5. GIVE THE ROBOT ITS RULES ---
HULIOT_SYSTEM_PROMPT = f"""
You are the expert Technical Manager for Huliot India.
    
OFFICIAL RULES (From PDFs):
{catalog_text}

YOUR DIARY (Things you learned day-by-day):
{diary_memory}
    
Answer questions using ONLY the PDFs and your Diary. Do not guess.
"""

# Use whatever brain you select in the dropdown
model = genai.GenerativeModel(
    model_name=selected_model,
    system_instruction=HULIOT_SYSTEM_PROMPT,
    generation_config={"temperature": 0.0}
)

# --- 6. CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. TALK TO THE ROBOT ---
user_question = st.chat_input("Type your question here...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # THE MAGIC TRICK: Teaching the robot!
    if user_question.startswith("LEARN:"):
        new_fact = user_question.replace("LEARN:", "").strip()
        with open("robot_diary.txt", "a", encoding="utf-8") as file:
            file.write("- " + new_fact + "\n")
        ai_answer = f"✍️ I just wrote this in my diary: '{new_fact}'. I will remember it tomorrow!"
    
    # NORMAL CHAT
    else:
        try:
            response = model.generate_content(user_question)
            ai_answer = response.text
        except Exception as e:
            ai_answer = f"⚠️ Oops! Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
