# Retrieval-Augmented Generation (RAG) System for Document Insights

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload documents, vectorize the content, and query specific information using a Large Language Model (LLM). The system is designed to provide precise information retrieval and interactive Q&A capabilities, leveraging semantic retrieval and scalable cloud services.

## Features
- **Document Ingestion:** Upload and preprocess documents for retrieval.
- **Vectorization:** Generate embeddings using OpenAI and store them in a FAISS vector database.
- **Semantic Retrieval:** Fetch document chunks using a conversational retrieval chain.
- **LLM Integration:** Use a language model for accurate and interactive question-answering.
- **Cloud Integration:** Azure services for indexing, storage, and scalability.

## How It Works
1. **Document Upload and Preprocessing:**
   - Documents are uploaded and split into chunks for efficient vectorization and retrieval.
2. **Vectorization:**
   - OpenAI embeddings are generated for document chunks and stored in a FAISS vector database.
3. **Query Mechanism:**
   - Users query the system using natural language, and relevant document chunks are retrieved.
   - An LLM processes these chunks to generate precise answers.
4. **System Evaluation:**
   - Performance is optimized for retrieval accuracy, scalability, and user experience.

## Technologies Used
- **Programming Languages:** Python
- **Libraries & Frameworks:** FAISS, OpenAI, Azure Cognitive Search
- **Cloud Services:** Azure Blob Storage, Azure OpenAI
- **Tools:** PyPDFLoader, CharacterTextSplitter, Conversational Retrieval Chain

## Example Code
```python
# Document Ingestion
loader = PyPDFLoader(file_path=pdf_file_path)
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Vectorization
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embedding_model)

# Retrieval Mechanism
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Querying
query = "What are the 15 CTE Industry Sectors in California?"
result = conversation_chain({'question': query})
print(result['answer'])
