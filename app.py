import streamlit as st
import google.generativeai as genai
import PyPDF2

# --- 1. SETUP GEMINI AI ---
# Paste your Huliot Streamlit Key inside the quotes below!
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. BUILD THE STREAMLIT WEBPAGE ---
st.set_page_config(page_title="Huliot AI Assistant", page_icon="💧")
st.title("💧 Huliot Technical Assistant")
st.write("Ask me anything about Huliot pipes, drainage systems, and acoustic solutions!")

with st.sidebar:
    st.header("🧠 Train your AI")
    st.write("Upload a Huliot PDF catalog to lock the AI into STRICT mode.")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    st.divider()
    st.header("⚙️ AI Settings")
    st.write("Select an AI Brain that has free credits left:")
    
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model = st.selectbox("Choose AI Brain", available_models)
    except Exception as e:
        st.error("Could not fetch models. Check API Key.")
        selected_model = "models/gemini-1.5-flash"

# --- READ THE PDF IF UPLOADED ---
catalog_text = ""
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        catalog_text += page.extract_text()
    st.sidebar.success("Catalog memorized! STRICT MODE ENABLED 🔒")

# --- 3. DUAL MODE SMART PROMPT ---
# This changes the AI's personality based on whether a file is uploaded!
if catalog_text == "":
    # MODE 1: NO FILE UPLOADED (Uses general internet knowledge)
    HULIOT_SYSTEM_PROMPT = """
    You are the expert Technical Manager for Huliot India.
    Please answer the user's questions about Huliot products, plumbing, and drainage systems using your general knowledge. 
    Keep your answers professional, helpful, and concise.
    """
else:
    # MODE 2: FILE UPLOADED (Strictly locked to the PDF only)
    HULIOT_SYSTEM_PROMPT = f"""
    You are the expert Technical Manager for Huliot India.
    CRITICAL INSTRUCTION: You must answer questions using ONLY the text provided in the OFFICIAL CATALOG DATA below. 
    DO NOT use your general AI internet knowledge. DO NOT guess. 
    If the exact answer cannot be found in the text below, you must reply exactly with: "I'm sorry, but that information is not in the uploaded catalog."

    OFFICIAL CATALOG DATA:
    {catalog_text}
    """

# Create the AI model
model = genai.GenerativeModel(
    model_name=selected_model,
    system_instruction=HULIOT_SYSTEM_PROMPT
)

# --- 4. CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. HANDLE NEW USER MESSAGES ---
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