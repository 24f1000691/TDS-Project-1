# query.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from rag_system import generate_rag_answer # <--- NEW IMPORT

app = FastAPI()

# Define the request model based on what promptfoo will send
class QueryRequest(BaseModel):
    question: str
    # You might also want to add link if promptfoo sends it consistently,
    # but the core RAG function usually just takes the question.
    # link: Optional[str] = None

# Endpoint for the RAG system
@app.post("/ask") # <--- RECOMMEND CHANGING BACK TO /ask for clarity and consistency
async def query_knowledge_base(request: QueryRequest):
    # Call your RAG function here
    rag_response = generate_rag_answer(request.question) # <--- Using request.question

    # Parse the rag_response to get answer and links if your RAG returns them separately
    # For now, let's assume generate_rag_answer returns a string.
    # You'll need to adapt this based on what generate_rag_answer actually returns (string or dict)

    # If generate_rag_answer returns a string (like the last code I gave for it):
    return JSONResponse(content={
        "answer": rag_response,
        "links": [] # If generate_rag_answer doesn't return links, you can leave this empty or infer
    })
    # If generate_rag_answer returns a dict like {"answer": "...", "links": [...]}
    # return JSONResponse(content=rag_response)


# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RAG API is running"}

