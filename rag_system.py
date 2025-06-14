import os
import tiktoken
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables (for local development)
load_dotenv()

# --- Configuration ---
# Use OPENAI_API_KEY for the actual OpenAI client initialization
# and ensure this is what's set in Vercel for OpenAI
# If API_KEY is specifically for AIPipe's authentication:
# AIPipe_API_KEY = os.getenv("API_KEY") # Consider renaming to AIPipe_API_KEY if distinct

# Ensure OPENAI_API_KEY is read correctly (either from system env or .env)
# The OpenAI client will *prefer* OPENAI_API_KEY from environment if not passed explicitly.
# Since you're passing it, we'll keep the name matching the passed variable.
OPENAI_API_KEY_FOR_CLIENT = os.getenv("API_KEY") # This maps Vercel's API_KEY to what OpenAI client expects for its api_key param
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Ensure all necessary environment variables are set
if not all([OPENAI_API_KEY_FOR_CLIENT, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    # This error message is for local testing. On Vercel, it might not print.
    raise ValueError("Missing one or more environment variables. Check your .env file or Vercel settings.")

# --- Initialize Pinecone Client (MOVED TO TOP TO BE DEFINED BEFORE USE) ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone client initialized successfully.") # For debugging
except Exception as e:
    print(f"Error initializing Pinecone client: {e}")
    raise

# Initialize OpenAI client, pointing to AIPipe's base URL
openai_client = OpenAI(
    api_key=OPENAI_API_KEY_FOR_CLIENT, # Use the correctly named variable
    base_url="https://aipipe.org/openai/v1" # This points to AIPipe
)
print("OpenAI client initialized successfully with AIPipe base URL.") # For debugging


# --- Constants ---
# IMPORTANT: These model names MUST BE AVAILABLE AND SUPPORTED BY AIPipe/AIProxy
# for their OpenAI-compatible proxy endpoint.
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL_VISION = "gpt-4o" # Confirm AIPipe supports this
LLM_MODEL_TEXT_ONLY = "gpt-3.5-turbo" # Confirm AIPipe supports this

MAX_TOKENS_FOR_LLM_VISION = 4096
RESERVED_QUERY_TOKENS = 500

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using OpenAI's text-embedding-3-small via AIPipe."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise

def retrieve_documents(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Retrieves the top_k most similar document chunks from Pinecone.
    """
    try:
        index = pc.Index(PINECONE_INDEX_NAME) # pc is now defined
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        retrieved_docs = []
        for match in query_results.matches:
            retrieved_docs.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get('text', ''),
                "title": match.metadata.get('title', 'No Title'),
                "url": match.metadata.get('url', '#')
            })
        return retrieved_docs
    except Exception as e:
        print(f"Error retrieving documents from Pinecone: {e}")
        raise

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string for a given model."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

def generate_llm_response(query: str, retrieved_documents: list[dict], image_data_list: list[str] = None) -> dict:
    """
    Generates a response using the LLM based on the user query, retrieved documents,
    and optional image data.
    Dynamically prunes documents to fit within the LLM's context window.
    """

    llm_model_to_use = LLM_MODEL_VISION if image_data_list else LLM_MODEL_TEXT_ONLY
    max_tokens_for_llm_effective = MAX_TOKENS_FOR_LLM_VISION if image_data_list else MAX_TOKENS_FOR_LLM_VISION

    system_prompt_template = """You are a helpful assistant that answers questions based on the provided text context and any given images.
If the answer is not available in the provided text context, politely state that you don't have enough information.
If images are provided, analyze them to help answer the question.
Avoid making up answers.

Context:"""

    context_str_parts = []
    sources = []

    current_tokens = num_tokens_from_string(system_prompt_template + query, llm_model_to_use) + RESERVED_QUERY_TOKENS
    
    for doc in retrieved_documents:
        doc_content = doc.get('text', '')
        doc_url = doc.get('url', '#')
        doc_title = doc.get('title', 'No Title')

        doc_for_llm = f"\n\n--- Document from {doc_title} (Source: {doc_url}) ---\n{doc_content}"
        doc_tokens = num_tokens_from_string(doc_for_llm, llm_model_to_use)

        if current_tokens + doc_tokens < max_tokens_for_llm_effective:
            context_str_parts.append(doc_for_llm)
            # Ensure only unique sources are added if desired, otherwise just append
            sources.append({"title": doc_title, "url": doc_url})
            current_tokens += doc_tokens
        else:
            break

    full_context_str = "\n".join(context_str_parts)

    messages = []
    messages.append({"role": "system", "content": system_prompt_template + full_context_str})

    user_content = []
    user_content.append({"type": "text", "text": query})

    if image_data_list:
        for base64_image in image_data_list:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "auto"
                }
            })
    
    messages.append({"role": "user", "content": user_content})

    try:
        response = openai_client.chat.completions.create(
            model=llm_model_to_use,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        llm_answer = response.choices[0].message.content.strip()
        # Ensure 'sources' key is present even if empty
        return {"answer": llm_answer, "sources": sources}
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        # Return a fallback answer and empty sources on error
        return {"answer": "I'm sorry, I encountered an error while trying to generate a response.", "sources": []}

# --- Main RAG Orchestration Function ---
def generate_rag_answer(user_query: str, image_data_list: list[str] = None) -> dict:
    """
    Orchestrates the RAG process: embed query, retrieve docs, generate LLM response.
    Accepts optional image data.
    """
    try:
        # 1. Embed the user query for retrieval
        query_embedding = get_embedding(user_query)

        # 2. Retrieve relevant documents from Pinecone
        retrieved_documents = retrieve_documents(query_embedding, top_k=7)

        # 3. Generate LLM response based on query, retrieved documents, and optional images
        llm_output = generate_llm_response(user_query, retrieved_documents, image_data_list)
        
        return llm_output
    except Exception as e:
        print(f"An error occurred in RAG process: {e}")
        # Return a consistent error message and empty sources in case of top-level RAG error
        return {"answer": "I'm sorry, an internal error occurred in the RAG system.", "sources": []}