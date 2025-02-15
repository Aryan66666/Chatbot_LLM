import os
import zipfile
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found! Set it in the .env file.")

app = FastAPI()

PERSIST_DIR = "docs/chroma/"
ZIP_FILE = "chroma.zip"

if not os.path.exists(PERSIST_DIR):
    if os.path.exists(ZIP_FILE):
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall("docs/")
        print("✅ Extracted chroma.zip successfully!")
    else:
        raise FileNotFoundError(f"❌ {ZIP_FILE} not found! Please upload the zip file.")

embedding = SentenceTransformerEmbeddings()
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,  
    model="deepseek/deepseek-r1-distill-llama-70b:free"
)

retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = qa_chain.run(request.question)
    return {"question": request.question, "answer": answer}
