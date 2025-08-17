import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def infer_state_from_query(query: str) -> str:
    """Naively infer a state slug from the user's query.

    Currently only supports California. If no state is found, defaults to
    "california".
    """
    query_lower = query.lower()
    if "california" in query_lower or "ca" in query_lower:
        return "california"
    # Default/fallback
    return "california"


def create_qa_chain(state_slug: str = "california"):
    """Load the vector store and set up the QA chain for a given state."""

    # Load environment variables from .env file
    load_dotenv()

    # Check if the OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set it in your .env file.")

    # 1. Load the local vector store for the selected state
    db_path = f"data/knowledge_base/faiss_index_{state_slug}"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # allow_dangerous_deserialization is needed for loading local FAISS indexes
    vector_store = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    print("Vector store loaded successfully.")

    # 2. Set up the LLM from OpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    print("OpenAI LLM loaded successfully.")

    # 3. Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    print("QA chain created.")
    return qa_chain


def get_response(query, chain):
    """Gets a response from the QA chain for a given query."""
    result = chain({"query": query})
    return result["result"]


if __name__ == "__main__":
    question = "I want to operate under a fictitious business name"
    print(f"\nQuery: {question}")

    slug = infer_state_from_query(question)
    qa_chain = create_qa_chain(slug)

    response = get_response(question, qa_chain)
    print(f"Answer: {response}")

    print("-" * 30)

    # question_2 = "What is a registered agent and do I need one in California?"
    # print(f"Query: {question_2}")
    # response_2 = get_response(question_2, qa_chain)
    # print(f"Answer: {response_2}")
    print("--- Source Chunks Used ---")
    for i, doc in enumerate(response["source_documents"]):
        print(f"--- Chunk {i+1} ---\n")
        print(doc.page_content)
        print("\n")

