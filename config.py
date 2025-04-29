# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- General Settings ---
PROJECT_NAME = "PolicyPal"

# --- Data Settings ---
# Directory where your source policy documents (PDFs) are stored
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# --- Text Splitting Settings ---
# Defines how documents are chunked for processing
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200 # Number of characters shared between consecutive chunks

# --- Embedding Model Settings ---
# Using OpenAI's embedding model
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Example model, check OpenAI docs for latest/best

# --- Vector Store Settings ---
# Directory to save the FAISS vector store index
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
VECTOR_STORE_INDEX_NAME = "policy_index" # Name for the FAISS index file

# --- LLM Settings ---
# Using OpenAI's chat model
LLM_MODEL_NAME = "gpt-3.5-turbo" # Example model, can use gpt-4 if preferred/available
LLM_TEMPERATURE = 0.1 # Controls randomness (0 = deterministic, 1 = more random)
LLM_MAX_TOKENS = 1024 # Max number of tokens the LLM should generate

# --- Retriever Settings ---
# Number of relevant document chunks to retrieve for context
RETRIEVER_K = 5 # Retrieve top 5 most similar chunks

# --- API Keys (Loaded from environment variables) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Validation ---
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in the .env file.")

if not os.path.exists(DOCUMENTS_DIR):
     print(f"Warning: Documents directory not found at {DOCUMENTS_DIR}. Please create it and add policy documents.")
     # Optionally, create the directory: os.makedirs(DOCUMENTS_DIR, exist_ok=True)

if not os.path.exists(VECTOR_STORE_DIR):
    print(f"Creating vector store directory at {VECTOR_STORE_DIR}")
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

print("Configuration loaded successfully.")