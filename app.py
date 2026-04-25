import streamlit as st
import google.generativeai as genai
import PyPDF2
import os
import glob

# --- 1. SETUP GEMINI AI ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. BUILD THE STREAMLIT WEBPAGE ---
st.set_page_config(page_title="Huliot AI Assistant", page_icon="💧")
st.title("💧 Huliot Technical Assistant")
st.write("Ask me anything about Huliot pipes, drainage systems, and acoustic solutions!")

# --- 3. AUTO-LOAD THE KNOWLEDGE BASE (MULTI-FILE) ---
# This @st.cache_data tag tells Streamlit to read the PDFs once when the app wakes up
@st.cache_data
def load_knowledge_base():
    text = ""
    # glob.glob("*.pdf") finds EVERY file ending in .pdf in your folder!
    pdf_files = glob.glob("*.pdf") 
    
    if len(pdf_files) == 0:
        return "", 0 # No files found
        
    for file_name in pdf_files:
        try:
            with open(file_name, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception:
            pass # If one file is corrupted, skip it and keep reading the others!
            
    return text, len(pdf_files)

catalog_text, file_count = load_knowledge_base()

with st.sidebar:
    st.header("🧠 AI Brain Status")
    if file_count > 0:
        st.success(f"Knowledge Base Loaded! ({file_count} files memorized)")
        st.write("STRICT MODE ENABLED 🔒")
    else:
        st.error("⚠️ Could not find any PDF files. Please upload them to GitHub.")
    
    st.divider()
    st.header("⚙️ AI Settings")
    st.write("Select an AI Brain that has free credits left:")
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model = st.selectbox("Choose AI Brain", available_models)
    except Exception as e:
        st.error("Could not fetch models. Check API Key.")
        selected_model = "models/gemini-1.5-flash"

# --- 4. STRICT PROMPT ---
if catalog_text != "":
    HULIOT_SYSTEM_PROMPT = f"""
    You are the expert Technical Manager for Huliot India.
    CRITICAL INSTRUCTION: You must answer questions using ONLY the text provided in the OFFICIAL CATALOG DATA below. 
    DO NOT use your general AI internet knowledge. DO NOT guess. 
    If the exact answer cannot be found in the text below, you must reply exactly with: "I'm sorry, but that information is not in the uploaded catalog."

    OFFICIAL CATALOG DATA:
    {catalog_text}
    """
else:
    # Fallback if the files break or are missing
    HULIOT_SYSTEM_PROMPT = "You are a helpful AI. Please answer questions based on your general knowledge."

model = genai.GenerativeModel(
    model_name=selected_model,
    system_instruction=HULIOT_SYSTEM_PROMPT,
    generation_config={"temperature": 0.0}
)

# --- 5. CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. HANDLE NEW USER MESSAGES ---
user_question = st.chat_input("Type your question here...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    try:
        response = model.generate_content(user_question)
        ai_answer = response.text
    except Exception as e:
        ai_answer = f"⚠️ Oops! This model threw an error: {e}"

    with st.chat_message("assistant"):
        st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
