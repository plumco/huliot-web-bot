import streamlit as st
import google.generativeai as genai
import PyPDF2
import os
import glob

# --- 1. SETUP GEMINI AI ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. BUILD THE WEBPAGE ---
st.set_page_config(page_title="Huliot Auto-Learning Agent", page_icon="🤖")
st.title("🤖 Huliot Autonomous Agent")
st.write("I learn automatically! Just talk to me or correct me, and I will update my own diary.")

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
        with st.expander("Peek inside the Diary"):
            st.write(diary_memory)
    else:
        st.warning("📓 Diary is empty right now.")
        
    st.divider()
    st.header("⚙️ AI Settings")
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model = st.selectbox("Choose AI Brain", available_models)
    except Exception as e:
        selected_model = "models/gemini-1.5-flash"

# --- 5. THE AGENT LEARNING FUNCTION (BACKGROUND BRAIN) ---
def think_and_learn(user_text, ai_text):
    """This runs secretly in the background to extract new rules!"""
    reflection_prompt = f"""
    You are the "Memory Manager" for an AI assistant. Analyze this short conversation:
    User said: "{user_text}"
    AI replied: "{ai_text}"
    
    Did the user explicitly teach a new rule, correct a mistake, or provide a specific site instruction that the AI should remember forever?
    - If YES: Extract the core fact into one single, clear sentence.
    - If NO (it's just a normal question, greeting, or the AI answered correctly): Reply exactly with the word NONE.
    """
    try:
        thinking_model = genai.GenerativeModel("models/gemini-1.5-flash")
        thought = thinking_model.generate_content(reflection_prompt).text.strip()
        
        # If the brain didn't say "NONE", it means it learned something new!
        if thought != "NONE" and thought != "" and "NONE" not in thought:
            with open("robot_diary.txt", "a", encoding="utf-8") as file:
                file.write("- " + thought + "\n")
            return thought 
    except Exception:
        pass
    return None

# --- 6. GIVE THE ROBOT ITS RULES ---
HULIOT_SYSTEM_PROMPT = f"""
You are the expert Technical Manager for Huliot India.
    
OFFICIAL RULES (From PDFs):
{catalog_text}

YOUR DIARY (Things you learned day-by-day):
{diary_memory}
    
Answer questions using ONLY the PDFs and your Diary. Be helpful and professional.
"""

model = genai.GenerativeModel(
    model_name=selected_model,
    system_instruction=HULIOT_SYSTEM_PROMPT,
    generation_config={"temperature": 0.0}
)

# --- 7. CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. TALK TO THE AGENT ---
user_question = st.chat_input("Ask a question or correct the bot...")

if user_question:
    # 1. Print User Message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # 2. Generate Normal Answer
    with st.chat_message("assistant"):
        try:
            response = model.generate_content(user_question)
            ai_answer = response.text
        except Exception as e:
            ai_answer = f"⚠️ Oops! Error: {e}"
        st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})

    # 3. BACKGROUND THINKING (The Self-Learning Magic!)
    with st.spinner("🤖 Agent is analyzing the conversation..."):
        new_rule = think_and_learn(user_question, ai_answer)
        if new_rule:
            st.success(f"**🧠 Self-Learning Triggered!** I realized you taught me something new. I just saved this to my permanent memory: *{new_rule}*")
