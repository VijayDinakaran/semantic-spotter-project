# rag_pipeline.py
import logging
import config  # Assuming config.py is in the same directory or accessible

# --- Core LangChain component imports ---
# Updated imports to use langchain_core directly to potentially avoid Pydantic conflicts
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.vectorstores import VectorStoreRetriever # More specific import

# --- OpenAI specific imports ---
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings # Needed for the __main__ block example

# --- Community imports (if needed, e.g., for FAISS in example) ---
from langchain_community.vectorstores import FAISS # Needed for the __main__ block example


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Prompt Templates ---

# Condensed Question Prompt (Optional - for conversation history, not used in basic RAG)
# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Answer Prompt (Core RAG Prompt)
# This template defines how the LLM should behave.
ANSWER_TEMPLATE = """
You are an expert assistant specialized in interpreting insurance policy documents.
Answer the question based *only* on the following context provided from the policy document(s).
If the context does not contain the information needed to answer the question, state clearly:
"Based on the provided documents, I cannot answer this question."
Do not add information that is not present in the context. Be concise and accurate.

Context:
{context}

Question:
{question}

Answer:
"""
# Create the prompt template instance from the string template.
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


def get_llm() -> ChatOpenAI:
    """
    Initializes and returns the ChatOpenAI LLM instance based on config.

    Returns:
        ChatOpenAI: An instance of the language model.

    Raises:
        ValueError: If the API key is not configured.
        Exception: For other initialization errors.
    """
    logger.info(f"Initializing LLM: {config.LLM_MODEL_NAME}")
    if not config.OPENAI_API_KEY:
        # Ensure API key is checked here as well, though config.py should raise it first.
        raise ValueError("OpenAI API Key not configured properly.")
    try:
        llm = ChatOpenAI(
            model_name=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            openai_api_key=config.OPENAI_API_KEY
            # Consider adding model_kwargs for specific needs if required
            # model_kwargs={"top_p": 0.9}
        )
        logger.info("LLM initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise


def format_docs(docs: list) -> str:
    """
    Helper function to format retrieved documents (list of Document objects)
    into a single string, separated by double newlines.

    Args:
        docs (list): A list of Langchain Document objects.

    Returns:
        str: A single string containing the page content of all documents.
    """
    if not docs:
        return "No context documents found."
    # Extracts the page_content attribute from each Document object in the list
    # and joins them together with two newline characters as a separator.
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever: VectorStoreRetriever, llm: ChatOpenAI):
    """
    Creates the complete RAG chain using LangChain Expression Language (LCEL).

    This chain orchestrates the flow:
    1. Takes a question as input.
    2. Uses the retriever to fetch relevant documents ('context').
    3. Passes the original question and the retrieved context to the prompt template.
    4. Sends the formatted prompt to the LLM.
    5. Parses the LLM's output into a string.

    Args:
        retriever (VectorStoreRetriever): The retriever instance configured
                                          to fetch relevant document chunks.
        llm (ChatOpenAI): The language model instance for generating answers.

    Returns:
        Runnable: The Langchain Runnable sequence representing the RAG chain.
    """
    logger.info("Creating RAG chain...")

    # Define the processing steps using LCEL (LangChain Expression Language)

    # Step 1: Parallel processing setup.
    # This dictionary defines inputs for the next step (the prompt).
    # 'context': Fetches documents using the retriever, then formats them using format_docs.
    # 'question': Simply passes the original input question through unchanged.
    # RunnableParallel executes these operations potentially in parallel.
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )

    # Step 2: Define the main chain sequence using the pipe operator (|).
    # - setup_and_retrieval: Provides the context and question.
    # - ANSWER_PROMPT: Formats the context and question into the final prompt for the LLM.
    # - llm: Sends the prompt to the language model.
    # - StrOutputParser: Extracts the string content from the LLM's response object.
    rag_chain = setup_and_retrieval | ANSWER_PROMPT | llm | StrOutputParser()

    logger.info("RAG chain created successfully.")
    return rag_chain

# Example usage block (optional, for testing this script directly)
# This block demonstrates how to use the functions if the script is run directly.
# It requires setting up a dummy retriever, which depends on having a vector store.
if __name__ == "__main__":
    # This part requires setting up a retriever first, which depends on
    # having a vector store. We'll simulate it with a dummy FAISS store.
    import numpy as np # Only needed for dummy example if used

    print("\n--- Testing RAG Chain Creation (with dummy retriever) ---")

    try:
        # Ensure necessary config is available for the test
        if not config.OPENAI_API_KEY:
             print("Error: OPENAI_API_KEY not found in config for testing.")
        else:
            # 1. Create a dummy FAISS index for testing structure
            print("Initializing dummy embedding model...")
            # Use the embedding model configured in config.py
            dummy_embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL_NAME,
                openai_api_key=config.OPENAI_API_KEY
            )

            print("Creating dummy vector store...")
            # Simple text examples to populate the dummy store
            dummy_texts = [
                "Policy Section A covers collision damage with a $500 deductible.",
                "Liability coverage limits are $100,000 per person.",
                "Flood damage is explicitly excluded under this standard policy.",
                "Rental car reimbursement is available as an optional add-on."
                ]
            # Create an in-memory FAISS store from these texts
            dummy_vectorstore = FAISS.from_texts(dummy_texts, dummy_embeddings)
            # Create a retriever from the dummy store
            dummy_retriever = dummy_vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2
            print("Dummy retriever created.")

            # 2. Get LLM instance
            print("Initializing LLM...")
            llm_instance = get_llm()
            print("LLM instance obtained.")

            # 3. Create the RAG chain using the dummy retriever and real LLM
            print("Creating RAG chain...")
            chain = create_rag_chain(dummy_retriever, llm_instance)
            print("RAG chain created successfully.")

            # 4. Test invocation (this will make an API call to OpenAI)
            print("\n--- Testing RAG Chain Invocation ---")
            test_query = "What is the deductible for collision?"
            # test_query = "Is flood damage covered?" # Another test query
            print(f"Query: {test_query}")

            # Use invoke for a single query/response
            response = chain.invoke(test_query)

            print("\n--- Response ---")
            print(response)

    except ValueError as ve:
        print(f"Configuration Error during test: {ve}")
    except ImportError as ie:
         print(f"Import Error during test setup: {ie}. Make sure all dependencies are installed.")
    except Exception as e:
        print(f"An unexpected error occurred during RAG chain testing: {e}")
        # Optionally print traceback for more detail
        # import traceback
        # traceback.print_exc()

