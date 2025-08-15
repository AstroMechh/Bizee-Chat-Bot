import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vector_db():
    """
    Reads text from a file, chunks it, creates embeddings,
    and stores them in a FAISS vector database.
    """
    # 1. Load the document
    loader = TextLoader("data/processed_text/california_llc_guide.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} document.")

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    # 3. Initialize the embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    print("Embeddings model loaded.")

    # 4. Create the FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created.")

    # 5. Save the vector store locally
    if not os.path.exists("data/knowledge_base"):
        os.makedirs("data/knowledge_base")
    vector_store.save_local("data/knowledge_base/faiss_index_california")
    print("Vector store saved locally at data/knowledge_base/faiss_index_california")

if __name__ == "__main__":
    create_vector_db()