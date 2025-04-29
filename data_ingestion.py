# data_ingestion.py
import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredFileLoader
from langchain.schema import Document
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents(source_directory: str = config.DOCUMENTS_DIR) -> List[Document]:
    """
    Loads documents from the specified directory using appropriate loaders.
    Currently supports PDF files.

    Args:
        source_directory (str): The path to the directory containing documents.

    Returns:
        List[Document]: A list of loaded Langchain Document objects.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        Exception: For errors during document loading.
    """
    if not os.path.isdir(source_directory):
        raise FileNotFoundError(f"Source directory '{source_directory}' not found.")

    logging.info(f"Loading documents from: {source_directory}")
    documents = []

    # Using DirectoryLoader with PyPDFLoader for simplicity for PDF files
    # Glob pattern looks for any PDF file recursively
    pdf_loader = DirectoryLoader(
        source_directory,
        glob="**/*.pdf", # Looks for PDF files in the directory and subdirectories
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True # Speeds up loading for multiple files
    )

    try:
        loaded_docs = pdf_loader.load()
        if not loaded_docs:
            logging.warning(f"No PDF documents found or loaded from {source_directory}")
        else:
            logging.info(f"Successfully loaded {len(loaded_docs)} PDF document(s).")
            documents.extend(loaded_docs)

        # You could add loaders for other file types here (e.g., .txt, .docx)
        # txt_loader = DirectoryLoader(source_directory, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
        # loaded_txt = txt_loader.load()
        # documents.extend(loaded_txt)

    except Exception as e:
        logging.error(f"An error occurred during document loading: {e}", exc_info=True)
        # Depending on requirements, you might want to raise the exception
        # or just return the documents loaded so far.
        # raise Exception(f"Failed to load documents: {e}") from e
        return documents # Return successfully loaded docs, if any

    logging.info(f"Total documents loaded: {len(documents)}")
    return documents

# Example usage (optional, for testing this script directly)
if __name__ == "__main__":
    try:
        # Ensure the documents directory exists for the test
        if not os.path.exists(config.DOCUMENTS_DIR):
             os.makedirs(config.DOCUMENTS_DIR)
             print(f"Created documents directory: {config.DOCUMENTS_DIR}")
             print("Please add some PDF files to this directory for testing.")

        docs = load_documents()
        if docs:
            print(f"\nLoaded {len(docs)} documents.")
            # Print info about the first document (if loaded)
            print("\n--- Sample Document Info ---")
            print(f"Content snippet (first 200 chars): {docs[0].page_content[:200]}")
            print(f"Metadata: {docs[0].metadata}")
        else:
            print("No documents were loaded. Ensure PDFs are in the 'documents' folder.")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as general_error:
        print(f"An unexpected error occurred: {general_error}")