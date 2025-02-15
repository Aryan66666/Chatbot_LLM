import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load .env file
load_dotenv()

# Get API key securely
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Set up the vector database
PERSIST_DIR = "docs/chroma/"
embedding = SentenceTransformerEmbeddings()
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

# Initialize LLM with the loaded API key
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,  # Use the environment variable
    model="deepseek/deepseek-r1-distill-llama-70b:free"
)

# Set up retriever and QA chain
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = qa_chain.run(request.question)
    return {"question": request.question, "answer": answer}
