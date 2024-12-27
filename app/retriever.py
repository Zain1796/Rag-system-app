from chromadb import Client
from chromadb.config import Settings
from langchain_community.embeddings import OpenAIEmbeddings
import os
from chromadb import PersistentClient

def initialize_chroma_db():
    """Initialize Chroma DB for vector storage."""
    client = PersistentClient(path="./db")  # Use the new PersistentClient
    return client


# def add_to_chroma(client, vectors, chunks):
#     collection = client.get_or_create_collection(name="my_collection")
    
#     # Generate unique IDs for each vector
#     ids = [f"doc_{i}" for i in range(len(vectors))]
    
#     # Add vectors, metadata, and IDs to the collection
#     collection.add(
#         embeddings=vectors,
#         metadatas=chunks,
#         ids=ids  # Pass the IDs here
#     )
def add_to_chroma(client, vectors, chunks):
    """Add vectors and metadata to ChromaDB."""
    collection = client.get_or_create_collection(name="my_collection")
    
    # Ensure metadata is a dictionary with plain string values
    metadatas = [{"content": str(chunk)} for chunk in chunks]
    
    # Generate unique IDs for the vectors
    ids = [f"doc_{i}" for i in range(len(vectors))]
    
    # Add vectors, metadata, and IDs to the collection
    collection.add(
        embeddings=vectors,
        metadatas=metadatas,  # Use dictionaries for metadata
        ids=ids
    )



# def query_chroma(client, query_text, n_results=3):
#     """Query Chroma DB to retrieve relevant chunks."""
#     embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
#     query_vector = embeddings.embed_query(query_text)
#     collection = client.get_collection(name="my_collection")
#     results = collection.query(
#         query_embeddings=[query_vector],
#         n_results=n_results
#     )
#     return results["metadatas"]
def query_chroma(client, query_text, n_results=3):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    query_vector = embeddings.embed_query(query_text)
    
    collection = client.get_collection(name="my_collection")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    print(f"Query results: {results}")  # Debug log
    return results.get("metadatas", [])