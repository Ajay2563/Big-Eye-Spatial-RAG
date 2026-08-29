import os
import dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse # Correct import
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. BRAIN SETUP - Make sure your key is inside the quotes
os.environ["GOOGLE_API_KEY"] = "AIzaSy..." 

print("Loading PDF and Building Brain...")
# Ensure this file name is 100% correct in your folder
loader = PyPDFLoader("3D Printing Manual_compressed.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 2. API SETUP
app = FastAPI()

from mangum import Mangum
handler = Mangum(app) # This allows AWS Lambda to run FastAPI

# This allows your mobile browser to talk to this script
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

# 3. THE WEBSITE ROUTE (This sends the AR Interface to your phone)
@app.get("/")
async def get_website():
    # Make sure index.html is in the same folder as app.py
    return FileResponse("index.html")

# 4. THE AI ROUTE (The logic that answers questions)
@app.post("/ask")
def ask_ai(data: Query):
    relevant_docs = db.similarity_search(data.question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Context: {context_text}\n\nQuestion: {data.question}\n\nAnswer:"
    response = llm.invoke(prompt)
    return {"answer": response.content}

if __name__ == "__main__":
    import uvicorn
    # This starts the server on your laptop
    uvicorn.run(app, host="0.0.0.0", port=8000)
