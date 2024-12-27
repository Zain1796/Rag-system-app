from retriever import initialize_chroma_db, add_to_chroma, query_chroma


def inspect_all_docs(client):
    """Inspect all stored documents."""
    collection = client.get_collection(name="my_collection")
    results = collection.peek()  # Retrieves a sample of stored documents
    print(f"Stored documents: {results}")
    return results

client = initialize_chroma_db()
inspect_all_docs(client)

