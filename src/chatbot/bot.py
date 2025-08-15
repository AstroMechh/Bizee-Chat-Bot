import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def create_qa_chain():
    """
    Loads the vector store and sets up the Question-Answering chain using OpenAI.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Check if the OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set it in your .env file.")

    # 1. Load the local vector store
    db_path = "data/knowledge_base/faiss_index_california"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    # allow_dangerous_deserialization is needed for loading local FAISS indexes
    vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")

    # 2. Set up the LLM from OpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    print("OpenAI LLM loaded successfully.")

    # 3. Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True
    )
    print("QA chain created.")
    return qa_chain

def get_response(query, chain):
    """
    Gets a response from the QA chain for a given query.
    """
    result = chain({'query': query})
    return result['result']

if __name__ == "__main__":
    qa_chain = create_qa_chain()

    question = "I want to operate under a fictitious business name"
    print(f"\nQuery: {question}")

    response = get_response(question, qa_chain)
    print(f"Answer: {response}")

    print("-" * 30)

    # question_2 = "What is a registered agent and do I need one in California?"
    # print(f"Query: {question_2}")
    # response_2 = get_response(question_2, qa_chain)
    # print(f"Answer: {response_2}")
    print("--- Source Chunks Used ---")
    for i, doc in enumerate(response['source_documents']):
        print(f"--- Chunk {i+1} ---\n")
        print(doc.page_content)
        print("\n")