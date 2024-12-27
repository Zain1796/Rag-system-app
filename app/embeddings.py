import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()

def vectorize_document(file_path):
    """Preprocess and vectorize a document."""
    # with open(file_path, 'r') as f:
    #     content = f.read()
    reader = PdfReader(file_path)
    content = ""
    for page in reader.pages:
        content += page.extract_text()
    # Split document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)

    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectors = embeddings.embed_documents(chunks)

    return vectors, chunks
