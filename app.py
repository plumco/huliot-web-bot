# --- 2. THE VECTOR DATABASE (SUPER MEMORY) ---
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

    # NEW FIX: Filter out completely blank pages that crash Google's API
    documents = [doc for doc in documents if doc.page_content.strip() != ""]

    # 2. Chop them into smaller, safer puzzle pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Turn text into Math (Embeddings) safely!
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=API_KEY,
            task_type="retrieval_document" # Tells Google exactly what we are doing
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db, len(pdf_files)
    except Exception as e:
        # If Google blocks us, show the actual error message on the screen instead of crashing!
        st.error(f"⚠️ Google API blocked the database creation. Error: {e}")
        return None, 0

with st.spinner("Building Vector Database... (This takes a few seconds)"):
    vector_db, file_count = build_vector_database()
