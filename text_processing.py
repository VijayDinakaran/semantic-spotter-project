# text_processing.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import config
import logging
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_documents(documents: List[Document],
                    chunk_size: int = config.CHUNK_SIZE,
                    chunk_overlap: int = config.CHUNK_OVERLAP) -> List[Document]:
    """
    Splits a list of Langchain Document objects into smaller chunks.

    Args:
        documents (List[Document]): The list of documents to split.
        chunk_size (int): The maximum number of characters per chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Document]: A list of chunked Langchain Document objects.
                       Each chunk inherits metadata from its parent document.
    """
    logging.info(f"Splitting {len(documents)} documents into chunks...")
    logging.info(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use standard character length
        add_start_index=True, # Adds metadata about where the chunk started in the original doc
    )

    try:
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Successfully split documents into {len(chunks)} chunks.")

        # Optional: Log info about the first chunk
        # if chunks:
        #     logging.debug(f"First chunk metadata: {chunks[0].metadata}")
        #     logging.debug(f"First chunk content snippet: {chunks[0].page_content[:100]}...")

    except Exception as e:
        logging.error(f"An error occurred during document splitting: {e}", exc_info=True)
        raise Exception(f"Failed to split documents: {e}") from e

    return chunks

# Example usage (optional, for testing this script directly)
if __name__ == "__main__":
    from data_ingestion import load_documents # Need to import loading function

    try:
        # Ensure the documents directory exists for the test
        if not os.path.exists(config.DOCUMENTS_DIR):
             os.makedirs(config.DOCUMENTS_DIR)
             print(f"Created documents directory: {config.DOCUMENTS_DIR}")
             print("Please add some PDF files to this directory for testing.")

        docs = load_documents()
        if docs:
            chunks = split_documents(docs)
            print(f"\nSuccessfully split {len(docs)} documents into {len(chunks)} chunks.")

            # Print info about the first chunk (if exists)
            if chunks:
                print("\n--- Sample Chunk Info ---")
                print(f"Content snippet (first 100 chars): {chunks[0].page_content[:100]}")
                print(f"Metadata: {chunks[0].metadata}")
        else:
            print("No documents loaded, cannot perform splitting.")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")