import os
import logging
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY") # Your AIPipe token
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# --- Configuration Constants ---
INDEX_NAME = "tds-rag-index"
EMBEDDING_MODEL = "text-embedding-3-small" # Must match model used in preprocess.py

# --- Initialize OpenAI Client (for query embeddings) ---
# Ensure this base_url is correct for AIPipe embeddings
openai_client_for_retrieval = OpenAI(api_key=OPENAI_API_KEY, base_url="https://aipipe.org/openai/v1")

# --- Initialize Pinecone Client ---
pc_retriever = None
pinecone_index_retriever = None

# --- Flag to track initialization status ---
retriever_initialized = False

def initialize_retriever_pinecone():
    global pc_retriever, pinecone_index_retriever, retriever_initialized
    if retriever_initialized: # Prevent re-initialization if already successful
        return True

    # Check for missing environment variables before attempting Pinecone connection
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        logger.error("PINECONE_API_KEY or PINECONE_ENVIRONMENT environment variables not set. Retriever cannot initialize.")
        return False # Return False instead of exit(1)

    try:
        pc_retriever = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        # Use .names() for list_indexes as per successful preprocess.py logic
        if INDEX_NAME not in pc_retriever.list_indexes().names():
            logger.error(f"Pinecone index '{INDEX_NAME}' not found. Run preprocess.py first to create it.")
            return False # Return False instead of exit(1)
        
        pinecone_index_retriever = pc_retriever.Index(INDEX_NAME)
        logger.info(f"Retriever connected to Pinecone index '{INDEX_NAME}'.")
        retriever_initialized = True # Mark as initialized
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone for retriever: {e}")
        retriever_initialized = False
        return False # Return False instead of exit(1)

def get_query_embedding(text: str) -> list:
    """Gets embedding for a given query text."""
    # Ensure client is initialized before using it
    if not retriever_initialized and not initialize_retriever_pinecone():
        logger.error("Retriever not initialized, cannot get query embedding.")
        return None
        
    try:
        response = openai_client_for_retrieval.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting query embedding: {e}")
        return None

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieves top_k relevant text chunks from Pinecone based on the query.
    Returns a list of dictionaries, each containing 'id', 'score', and 'metadata'.
    """
    if not retriever_initialized and not initialize_retriever_pinecone():
        logger.error("Retriever not initialized, cannot retrieve relevant chunks.")
        return []

    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        return []

    try:
        query_results = pinecone_index_retriever.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True 
        )
        
        relevant_chunks = []
        for match in query_results.matches:
            relevant_chunks.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("combined_text", ""), 
                "source": match.metadata.get("source", "unknown"),
                "title": match.metadata.get("title", ""),
                "topic_title": match.metadata.get("topic_title", "")
            })
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query: '{query[:50]}...'")
        return relevant_chunks

    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return []

# Initialize Pinecone when the module is imported
# This call now sets the retriever_initialized flag and returns, it doesn't exit.
initialize_retriever_pinecone()