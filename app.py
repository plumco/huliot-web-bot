import streamlit as st
import os
import glob
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. SETUP ---
st.set_page_config(page_title="Huliot Super-Agent", page_icon="🤖")
st.title("🤖 Huliot Vector Agent")
st.write("Powered by a LangChain Vector Database (Like NotebookLM!)")

# Get API Key
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 2. THE VECTOR DATABASE ---
@st.cache_resource
def build_vector_database():
    pdf_files = glob.glob("*.pdf") 
    if len(pdf_files) == 0:
        return None, 0

    documents = []
    for file in pdf_files:
        try:
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
        except Exception:
            pass

    documents = [doc for doc in documents if doc.page_content.strip() != ""]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    try:
        # THE FIX: Going back to the reliable older model your API key supports!
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=API_KEY,
            task_type="retrieval_document"
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db, len(pdf_files)
    except Exception as e:
        st.error(f"⚠️ Google API blocked the database creation. Error: {e}")
        return None, 0

with st.spinner("Building Vector Database... (This takes a few seconds)"):
    vector_db, file_count = build_vector_database()

# --- 3. DIARY & SIDEBAR SETTINGS ---
diary_memory = ""
if os.path.exists("robot_diary.txt"):
    with open("robot_diary.txt", "r", encoding="utf-8") as file:
        diary_memory = file.read()

with st.sidebar:
    st.header("🧠 AI Brain Status")
    if vector_db:
        st.success(f"Vector Database Active! ({file_count} PDFs Indexed)")
    else:
        st.error("No PDFs found to build database.")
        
    if diary_memory != "":
        with st.expander("📓 Peek inside the Diary"):
            st.write(diary_memory)
            
    st.divider()
    
    # THE FIX: Bringing back your dropdown menu so you can pick a working brain!
    st.header("⚙️ AI Settings")
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model = st.selectbox("Choose AI Brain", available_models)
    except Exception:
        selected_model = "models/gemini-1.5-flash" # Fallback

# --- 4. ENGINE & LEARNING ---
def get_answer(question, chosen_model):
    if vector_db:
        search_results = vector_db.similarity_search(question, k=4)
        context_text = "\n\n".join([doc.page_content for doc in search_results])
    else:
        context_text = "No PDF context available."

    prompt = f"""You are the expert Technical Manager for Huliot India.
    SEARCH RESULTS: {context_text}
    YOUR DIARY: {diary_memory}
    QUESTION: {question}
    Answer professionally using ONLY the Search Results and your Diary."""
    
    try:
        # Uses the brain you pick in the dropdown!
        model = genai.GenerativeModel(chosen_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def auto_learn(user_txt, ai_txt, chosen_model):
    prompt = f"Did the user teach a new rule here?\nUser: {user_txt}\nAI: {ai_txt}\nIf YES, reply with ONE sentence. If NO, reply 'NONE'."
    try:
        model = genai.GenerativeModel(chosen_model)
        thought = model.generate_content(prompt).text.strip()
        if thought != "NONE" and thought != "" and "NONE" not in thought:
            with open("robot_diary.txt", "a", encoding="utf-8") as f:
                f.write("- " + thought + "\n")
            return thought
    except:
        pass
    return None

# --- 5. CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about Huliot...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            ai_answer = get_answer(user_question, selected_model)
        st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})

    with st.spinner("🤖 Analyzing conversation for new rules..."):
        new_rule = auto_learn(user_question, ai_answer, selected_model)
        if new_rule:
            st.success(f"**🧠 Self-Learning:** I saved this to my memory: *{new_rule}*")
