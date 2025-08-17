import os
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def create_vector_dbs():
    """Create a FAISS vector store for each processed state file."""
    processed_dir = "data/processed_text"
    kb_root = "data/knowledge_base"
    os.makedirs(kb_root, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs
    )

    for file_path in glob(os.path.join(processed_dir, "*_llc_content.txt")):
        state_slug = os.path.basename(file_path).replace("_llc_content.txt", "")
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, embeddings)

        save_path = os.path.join(kb_root, state_slug)
        vector_store.save_local(save_path)
        print(f"Vector store saved for {state_slug} at {save_path}")


if __name__ == "__main__":
    create_vector_dbs()
