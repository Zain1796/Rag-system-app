from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from app.embeddings import vectorize_document
from app.retriever import initialize_chroma_db, add_to_chroma, query_chroma
from app.llm import query_llm
import os
from dotenv import load_dotenv
import os

app = FastAPI()

# Initialize ChromaDB client globally
client = initialize_chroma_db()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG system API!"}

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    # Ensure the 'docs' directory exists
    os.makedirs("docs", exist_ok=True)

    # Save the uploaded file
    file_path = f"docs/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Vectorize the document
    vectors, chunks = vectorize_document(file_path)
    print("The document is vectorized and converted to chunks")

    # Add to ChromaDB
    add_to_chroma(client, vectors, chunks)

    return {"message": "Document processed and indexed successfully."}



@app.post("/query/")
async def ask_question(question: str = Form(...)):
    try:
        # Query ChromaDB
        results = query_chroma(client, question, n_results=3)
        print(f"Raw query results: {results}")  # Debug log
        
        # Extract context
        context = " ".join([item["content"] for item in results if "content" in item])
        print(f"Extracted context: {context}")  # Debug log
        
        if not context:
            raise ValueError("No valid content found in ChromaDB results.")
        
        # Query the LLM
        answer = query_llm(question, context)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

