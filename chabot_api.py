# -*- coding: utf-8 -*-
import os
import zipfile
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

PERSIST_DIR = "docs/chroma/"
ZIP_FILE = "chroma.zip"

# Ensure Chroma DB exists
if not os.path.exists(PERSIST_DIR):
    if os.path.exists(ZIP_FILE):
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall("docs/")
        print("✅ Extracted chroma.zip successfully!")
    else:
        raise FileNotFoundError(f"❌ {ZIP_FILE} not found! Please upload the zip file.")

embedding = SentenceTransformerEmbeddings()
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

# 🔹 Load Phi-2 model (Hugging Face)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 🔹 Create a text-generation pipeline
text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 🔹 Wrap it with HuggingFacePipeline for LangChain
llm = HuggingFacePipeline(pipeline=text_pipeline)


# 🔹 Set up the RetrievalQA chain correctly
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

# Define request model for FastAPI
class QuestionRequest(BaseModel):
    question: str

# FastAPI POST route to handle the question and answer
@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Get the answer from the RetrievalQA chain
    answer = qa_chain.run(request.question)

    return {"question": request.question, "answer": answer}
