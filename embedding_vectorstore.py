# embedding_vectorstore.py
import os
from typing import List
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initializes and returns the OpenAI embedding model.

    Returns:
        OpenAIEmbeddings: An instance of the embedding model.
    """
    logging.info(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}")
    try:
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY
        )
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        raise

def create_vector_store(chunks: List[Document],
                          embeddings: OpenAIEmbeddings,
                          index_path: str = os.path.join(config.VECTOR_STORE_DIR, config.VECTOR_STORE_INDEX_NAME)) -> FAISS:
    """
    Creates a FAISS vector store from document chunks and saves it to disk.

    Args:
        chunks (List[Document]): The list of document chunks.
        embeddings (OpenAIEmbeddings): The embedding model instance.
        index_path (str): The path where the FAISS index should be saved.

    Returns:
        FAISS: The created FAISS vector store instance.

    Raises:
        Exception: If vector store creation or saving fails.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty list of chunks.")

    logging.info(f"Creating FAISS vector store with {len(chunks)} chunks...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        logging.info("Vector store created successfully in memory.")

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        vector_store.save_local(index_path)
        logging.info(f"Vector store saved locally at: {index_path}")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to create or save vector store: {e}", exc_info=True)
        raise Exception(f"Vector store creation/saving failed: {e}") from e


def load_vector_store(index_path: str = os.path.join(config.VECTOR_STORE_DIR, config.VECTOR_STORE_INDEX_NAME),
                        embeddings: OpenAIEmbeddings = None) -> FAISS:
    """
    Loads an existing FAISS vector store from disk.

    Args:
        index_path (str): The path to the saved FAISS index folder.
        embeddings (OpenAIEmbeddings, optional): The embedding model instance.
                                                 If None, it initializes a new one.

    Returns:
        FAISS: The loaded FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index path doesn't exist.
        Exception: If loading fails.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector store index not found at: {index_path}. Please build it first.")

    if embeddings is None:
        embeddings = get_embedding_model() # Initialize if not provided

    logging.info(f"Loading vector store from: {index_path}")
    try:
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True # Required for FAISS loading
        )
        logging.info("Vector store loaded successfully.")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to load vector store: {e}", exc_info=True)
        raise Exception(f"Vector store loading failed: {e}") from e

# Example usage (optional, for testing this script directly)
if __name__ == "__main__":
    from data_ingestion import load_documents
    from text_processing import split_documents

    # --- Configuration for Test ---
    TEST_INDEX_PATH = os.path.join(config.VECTOR_STORE_DIR, "test_index")

    try:
        # 1. Setup: Ensure documents dir exists, load and split docs
        if not os.path.exists(config.DOCUMENTS_DIR):
            os.makedirs(config.DOCUMENTS_DIR)
            print(f"Created documents directory: {config.DOCUMENTS_DIR}")
            print("Please add some PDF files to this directory for testing.")

        docs = load_documents()
        if not docs:
            print("No documents found. Cannot proceed with vector store test.")
        else:
            chunks = split_documents(docs)
            if not chunks:
                 print("No chunks created. Cannot proceed.")
            else:
                # 2. Get Embeddings
                print("\n--- Getting Embedding Model ---")
                embeddings_model = get_embedding_model()
                print(f"Embedding model type: {type(embeddings_model)}")

                # 3. Create Vector Store
                print("\n--- Creating Vector Store ---")
                vectorstore = create_vector_store(chunks, embeddings_model, index_path=TEST_INDEX_PATH)
                print(f"Vector store created. Index size: {vectorstore.index.ntotal}")

                # 4. Load Vector Store
                print("\n--- Loading Vector Store ---")
                loaded_vectorstore = load_vector_store(index_path=TEST_INDEX_PATH, embeddings=embeddings_model)
                print(f"Vector store loaded. Index size: {loaded_vectorstore.index.ntotal}")

                # 5. Perform a similarity search (Example)
                print("\n--- Testing Similarity Search ---")
                query = "What is the coverage limit?"
                results = loaded_vectorstore.similarity_search(query, k=2)
                print(f"Found {len(results)} results for query: '{query}'")
                if results:
                    print("--- Sample Result 1 ---")
                    print(f"Content snippet: {results[0].page_content[:200]}...")
                    print(f"Metadata: {results[0].metadata}")

                # Clean up the test index
                # import shutil
                # if os.path.exists(TEST_INDEX_PATH):
                #     shutil.rmtree(TEST_INDEX_PATH)
                #     print(f"\nCleaned up test index at: {TEST_INDEX_PATH}")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except ValueError as val_error:
        print(f"Error: {val_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")