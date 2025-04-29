How to Run:

1. Save Files: Save each code block into its corresponding file within the policy_pal directory structure.
2. Create Folders: Manually create the documents and vector_store subdirectories inside policy_pal.
3. Add Documents: Place your public domain insurance policy PDF files into the documents folder.
4. Set API Key: Edit the .env file and replace "YOUR_API_KEY" with your actual OpenAI API key.
5. Install Dependencies: Open your terminal, navigate inside the policy_pal directory, and run:

pip install -r requirements.txt

or:

1. Copy/Download the repo and load it in vscode
2. create a .env file and add your OPENAI_API_KEY
3. Install dependecies


RUN A QUERY

First run will build the index (can take time depending on document size/count)
python main.py "What is the policy deductible for collision?"

Subsequent runs will load the existing index (faster setup)
python main.py "Are rental cars covered?"

Force rebuild the index if documents change
python main.py --rebuild "List the main exclusions."

Enter interactive mode
python main.py --interactive

Example Runs:


(base) vijaymallepudi@Vijays-MacBook-Pro Semantic_Spotter_Project % python main.py "Explain the HIV cover
age"
Configuration loaded successfully.
2025-04-29 21:55:02,088 - INFO - Setting up PolicyPal RAG system...
2025-04-29 21:55:02,088 - INFO - Initializing embedding model: text-embedding-3-small
2025-04-29 21:55:02,120 - INFO - Loading existing vector store from /Users/vijaymallepudi/Documents/Semantic_Spotter_Project/vector_store/policy_index...
2025-04-29 21:55:02,120 - INFO - Loading vector store from: /Users/vijaymallepudi/Documents/Semantic_Spotter_Project/vector_store/policy_index
2025-04-29 21:55:02,121 - INFO - Loading faiss.
2025-04-29 21:55:02,133 - INFO - Successfully loaded faiss.
2025-04-29 21:55:02,135 - INFO - Failed to load GPU Faiss: name 'GpuIndexIVFFlat' is not defined. Will not load constructor refs for GPU indexes.
2025-04-29 21:55:02,138 - INFO - Vector store loaded successfully.
2025-04-29 21:55:02,138 - INFO - Vector store loaded successfully.
2025-04-29 21:55:02,138 - INFO - Initializing LLM: gpt-3.5-turbo
2025-04-29 21:55:02,151 - INFO - LLM initialized successfully.
2025-04-29 21:55:02,151 - INFO - Retriever configured to fetch top 5 chunks.
2025-04-29 21:55:02,151 - INFO - Creating RAG chain...
2025-04-29 21:55:02,151 - INFO - RAG chain created successfully.
2025-04-29 21:55:02,151 - INFO - System setup completed in 0.06 seconds.
2025-04-29 21:55:02,151 - INFO - Processing query: Explain the HIV coverage
2025-04-29 21:55:03,304 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-04-29 21:55:04,499 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

--- Query ---
Explain the HIV coverage

--- Answer ---
Any sexually transmitted disease, or any condition related to HIV or AIDS is not covered under the insurance policy.
2025-04-29 21:55:04,521 - INFO - Query processed in 2.37 seconds.






(base) vijaymallepudi@Vijays-MacBook-Pro Semantic_Spotter_Project % python main.py "Explain Maturity Benefit "
Configuration loaded successfully.
2025-04-29 21:56:44,131 - INFO - Setting up PolicyPal RAG system...
2025-04-29 21:56:44,131 - INFO - Initializing embedding model: text-embedding-3-small
2025-04-29 21:56:44,174 - INFO - Loading existing vector store from /Users/vijaymallepudi/Documents/Semantic_Spotter_Project/vector_store/policy_index...
2025-04-29 21:56:44,174 - INFO - Loading vector store from: /Users/vijaymallepudi/Documents/Semantic_Spotter_Project/vector_store/policy_index
2025-04-29 21:56:44,175 - INFO - Loading faiss.
2025-04-29 21:56:44,198 - INFO - Successfully loaded faiss.
2025-04-29 21:56:44,202 - INFO - Failed to load GPU Faiss: name 'GpuIndexIVFFlat' is not defined. Will not load constructor refs for GPU indexes.
2025-04-29 21:56:44,207 - INFO - Vector store loaded successfully.
2025-04-29 21:56:44,207 - INFO - Vector store loaded successfully.
2025-04-29 21:56:44,207 - INFO - Initializing LLM: gpt-3.5-turbo
2025-04-29 21:56:44,220 - INFO - LLM initialized successfully.
2025-04-29 21:56:44,220 - INFO - Retriever configured to fetch top 5 chunks.
2025-04-29 21:56:44,220 - INFO - Creating RAG chain...
2025-04-29 21:56:44,221 - INFO - RAG chain created successfully.
2025-04-29 21:56:44,221 - INFO - System setup completed in 0.09 seconds.
2025-04-29 21:56:44,221 - INFO - Processing query: Explain Maturity Benefit 
2025-04-29 21:56:45,745 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-04-29 21:56:47,307 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

--- Query ---
Explain Maturity Benefit 

--- Answer ---
The Maturity Benefit is the benefit paid to the Policyholder on the Policy Maturity Date, provided the Policy remains in force and the Life Assured survives until the Policy Maturity Date. The amount of the Maturity Benefit depends on the Guaranteed Benefit Option chosen by the Policyholder as specified in the Policy Schedule.
2025-04-29 21:56:47,321 - INFO - Query processed in 3.10 seconds.
(base) vijaymallepudi@Vijays-MacBook-Pro Semantic_Spotter_Project % 
