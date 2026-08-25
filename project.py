import os
# We switch to HuggingFace for Embeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. SETUP - You only need the Gemini Key now
load_dotenv() # This loads your key from the .env file automatically

FILE_NAME = "3D Printing Manual_compressed.pdf"

if not os.path.exists(FILE_NAME):
    print(f"Error: {FILE_NAME} not found.")
else:
    print("PDF Found! Loading Local Embedding Model (Free)...")

    # 2. PREPARE THE DATA
    loader = PyPDFLoader(FILE_NAME)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. CREATE THE DATABASE (LOCAL - No API Key needed for this!)
    # This downloads a small model (~100MB) to your computer to handle the search
    print("Creating search index on your computer...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(chunks, embeddings)

    # 4. THE BRAIN (Google Gemini for answering)
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7
)

    def ask_the_manual(question):
        # Search the PDF using your local computer
        relevant_docs = db.similarity_search(question, k=3)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""
        You are a technical assistant. Use the following technical context to answer the question.
        
        CONTEXT:
        {context_text}
        
        QUESTION: 
        {question}
        
        ANSWER:"""
        
        response = llm.invoke(prompt)
        return response.content

    # --- RUN THE SYSTEM ---
    print("\n--- The Big Eye Brain is Ready! ---")
    print("Note: The first search might take 10 seconds to load the model.")
    
    while True:
        user_query = input("\nAsk a question (or 'exit'): ")
        if user_query.lower() == "exit":
            break
            
        print("Searching and thinking...")
        result = ask_the_manual(user_query)
        print(f"\nAI Assistant: {result}")