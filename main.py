# main.py
import os
import argparse
import logging
import time
import config # Load configuration settings
from data_ingestion import load_documents
from text_processing import split_documents
from embedding_vectorstore import get_embedding_model, create_vector_store, load_vector_store
from rag_pipeline import get_llm, create_rag_chain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_vector_store(force_rebuild=False):
    """
    Loads or creates the vector store.

    Args:
        force_rebuild (bool): If True, rebuilds the vector store even if it exists.

    Returns:
        FAISS: The loaded or newly created vector store instance.
    """
    index_path = os.path.join(config.VECTOR_STORE_DIR, config.VECTOR_STORE_INDEX_NAME)
    embeddings = get_embedding_model()

    if not force_rebuild and os.path.exists(index_path):
        logger.info(f"Loading existing vector store from {index_path}...")
        try:
            vector_store = load_vector_store(index_path=index_path, embeddings=embeddings)
            logger.info("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}. Rebuilding...")

    # Rebuild required or loading failed
    logger.info("Building new vector store...")
    if not os.path.exists(config.DOCUMENTS_DIR) or not os.listdir(config.DOCUMENTS_DIR):
         logger.error(f"Documents directory '{config.DOCUMENTS_DIR}' is empty or does not exist.")
         logger.error("Please add policy documents (PDFs) to the 'documents' folder.")
         raise FileNotFoundError(f"No documents found in {config.DOCUMENTS_DIR}")

    try:
        start_time = time.time()
        logger.info("Loading documents...")
        documents = load_documents()
        logger.info("Splitting documents into chunks...")
        chunks = split_documents(documents)
        logger.info("Creating vector store (this may take some time)...")
        vector_store = create_vector_store(chunks, embeddings, index_path=index_path)
        end_time = time.time()
        logger.info(f"Vector store built and saved successfully in {end_time - start_time:.2f} seconds.")
        return vector_store
    except Exception as e:
        logger.error(f"Fatal error during vector store setup: {e}", exc_info=True)
        raise # Propagate the error to stop execution

def main():
    """
    Main function to run the PolicyPal Q&A system.
    Handles argument parsing, sets up the RAG chain, and processes queries.
    """
    parser = argparse.ArgumentParser(description=f"{config.PROJECT_NAME} - Insurance Policy Q&A")
    parser.add_argument("query", type=str, nargs='?', help="The question to ask about the insurance policies.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vector store index.")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode to ask multiple questions.")

    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("You must provide a query or use --interactive mode.")
        return # Exit if no query and not interactive

    try:
        # --- Setup Phase ---
        logger.info("Setting up PolicyPal RAG system...")
        start_setup_time = time.time()

        # 1. Load or Build Vector Store
        vector_store = setup_vector_store(force_rebuild=args.rebuild)

        # 2. Initialize LLM
        llm = get_llm()

        # 3. Create Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
        logger.info(f"Retriever configured to fetch top {config.RETRIEVER_K} chunks.")

        # 4. Create RAG Chain
        rag_chain = create_rag_chain(retriever, llm)

        end_setup_time = time.time()
        logger.info(f"System setup completed in {end_setup_time - start_setup_time:.2f} seconds.")

        # --- Query Phase ---
        if args.interactive:
            print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.")
            while True:
                try:
                    user_query = input("\nAsk a question about the policy: ")
                    if user_query.lower() in ['exit', 'quit']:
                        break
                    if not user_query:
                        continue

                    start_query_time = time.time()
                    logger.info(f"Processing query: {user_query}")
                    result = rag_chain.invoke(user_query)
                    end_query_time = time.time()

                    print("\n--- Answer ---")
                    print(result)
                    logger.info(f"Query processed in {end_query_time - start_query_time:.2f} seconds.")

                except EOFError: # Handle Ctrl+D
                    break
                except KeyboardInterrupt: # Handle Ctrl+C
                    print("\nExiting interactive mode.")
                    break
            print("\nExited PolicyPal.")

        else:
            # Single query mode
            start_query_time = time.time()
            logger.info(f"Processing query: {args.query}")
            result = rag_chain.invoke(args.query)
            end_query_time = time.time()

            print("\n--- Query ---")
            print(args.query)
            print("\n--- Answer ---")
            print(result)
            logger.info(f"Query processed in {end_query_time - start_query_time:.2f} seconds.")

    except FileNotFoundError as e:
         logger.error(f"Setup Error: {e}")
         print(f"\nError: {e}. Please ensure documents are placed correctly and try again.")
    except ValueError as e: # Catch API key errors etc. from config
        logger.error(f"Configuration Error: {e}")
        print(f"\nConfiguration Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred. Check logs for details. Error: {e}")

if __name__ == "__main__":
    main()