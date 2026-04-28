import streamlit as st
import os
import glob
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# --- 1. SETUP ---
st.set_page_config(page_title="Huliot Super-Agent", page_icon="🤖")
st.title("🤖 Huliot Vector Agent")
st.write("Powered by a LangChain Vector Database (Like NotebookLM!)")

# Get API Key and set it for both standard Gemini and LangChain
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 2. THE VECTOR DATABASE (SUPER MEMORY) ---
# We use @st.cache_resource so it only builds the database ONCE to save speed.
@st.cache_resource
def build_vector_database():
    pdf_files = glob.glob("*.pdf") 
    if len(pdf_files) == 0:
        return None, 0

    documents = []
    # 1. Load all PDFs
    for file in pdf_files:
        try:
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
        except Exception:
            pass

    # 2. Chop them into smart "Puzzle Pieces" (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Turn text into Math (Embeddings) and build the Database!
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    return vector_db, len(pdf_files)

with st.spinner("Building Vector Database... (This takes a few seconds)"):
    vector_db, file_count = build_vector_database()

# --- 3. THE DIARY ---
diary_memory = ""
if os.path.exists("robot_diary.txt"):
    with open("robot_diary.txt", "r", encoding="utf-8") as file:
        diary_memory = file.read()

# --- 4. SIDEBAR STATUS ---
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
    # The Chat Brain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# --- 5. THE SEARCH & ANSWER ENGINE ---
def get_answer(question):
    # 1. Search the Database for the exact paragraphs needed
    if vector_db:
        search_results = vector_db.similarity_search(question, k=4) # Grab top 4 chunks
        context_text = "\n\n".join([doc.page_content for doc in search_results])
    else:
        context_text = "No PDF context available."

    # 2. Combine the Search Results, the Diary, and the Question
    prompt = f"""
    You are the expert Technical Manager for Huliot India.
    
    SEARCH RESULTS (From Database):
    {context_text}
    
    YOUR DIARY (Site Rules):
    {diary_memory}
    
    QUESTION: {question}
    
    Answer the question professionally using ONLY the Search Results and your Diary.
    If the answer is not in those sources, say: "I'm sorry, I don't have that in my database yet."
    """
    
    # 3. Ask Gemini to read the combined prompt
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 6. BACKGROUND AUTOLOOP (LEARNING) ---
def auto_learn(user_txt, ai_txt):
    prompt = f"""Did the user teach a new rule or correct a mistake here? 
    User: {user_txt}
    AI: {ai_txt}
    If YES, reply with ONE sentence summarizing the new rule. If NO, reply 'NONE'."""
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        thought = model.generate_content(prompt).text.strip()
        if thought != "NONE" and thought != "" and "NONE" not in thought:
            with open("robot_diary.txt", "a", encoding="utf-8") as f:
                f.write("- " + thought + "\n")
            return thought
    except:
        pass
    return None

# --- 7. CHAT INTERFACE ---
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

    # Generate answer using the Vector DB search!
    with st.chat_message("assistant"):
        ai_answer = get_answer(user_question)
        st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})

    # Run auto-learning in background
    with st.spinner("🤖 Analyzing..."):
        new_rule = auto_learn(user_question, ai_answer)
        if new_rule:
            st.success(f"**🧠 Self-Learning:** Saved to memory: *{new_rule}*")
