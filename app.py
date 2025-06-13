from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel, Field # type: ignore
from typing import List, Optional, Dict
import uvicorn # type: ignore
import os

# Import the RAG orchestration function
# Make sure this name matches the function in rag_system.py
from rag_system import generate_rag_answer # Changed from get_rag_answer

# Initialize FastAPI app
app = FastAPI(
    title="TDS Multimodal RAG API",
    description="A Retrieval Augmented Generation API for TDS and Discourse content, supporting text and images."
)

# Pydantic model for the incoming request body
class QueryRequest(BaseModel):
    question: str
    image_data: Optional[List[str]] = Field(
        None,
        alias="image",
        description="Optional list of Base64 encoded image data"
    )

# Pydantic model for the outgoing response body
class Source(BaseModel):
    title: str
    url: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.get("/")
async def read_root():
    return {"message": "Welcome to the TDS Multimodal RAG API. Use /api/ to get answers."}

@app.post("/api/", response_model=RAGResponse)
async def ask_question(request: QueryRequest):
    """
    Receives a student question and optional Base64 encoded image attachments,
    performs multimodal RAG, and returns an answer with sources.
    """
    try:
        user_question = request.question
        image_list_for_rag = []
        if request.image_data:
            for img_b64 in request.image_data:
                if img_b64.startswith("data:image/"):
                    image_list_for_rag.append(img_b64.split(",", 1)[1])
                else:
                    image_list_for_rag.append(img_b64)

        # Call the main RAG function, using the CORRECT imported name
        rag_output = generate_rag_answer(user_question, image_data_list=image_list_for_rag) # Changed from get_rag_answer
        
        return RAGResponse(answer=rag_output["answer"], sources=rag_output["sources"])
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

# --- For Local Development ---
if __name__ == "__main__":
    print("To run locally, use: uvicorn app:app --reload --host 0.0.0.0 --port 8000")
