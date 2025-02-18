from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import zipfile
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM   # Correct model type for T5
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA


app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow all domains to access your API

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

# Initialize the tokenizer and set the pad_token if not already set
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Initialize the model for sequence-to-sequence tasks
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# Resize token embeddings if the tokenizer has been updated
model.resize_token_embeddings(len(tokenizer))

# Initialize embeddings with the model
embedding = SentenceTransformerEmbeddings()

# Initialize the HuggingFace text generation model
generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# Create the HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generator)

# Initialize the Chroma vector store with the pre-existing persisted data
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

# Define the API route for question answering
@app.route('/ask', methods=['POST'])
def ask_question():
    # Get the question from the API request
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Use the loaded vector store and create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )

    # Get the result from the QA chain
    result = qa_chain({"query": question})  # Changed from invoke() to run()

    return jsonify({"result": result['result']})


if __name__ == '__main__':
    app.run(debug=False)
