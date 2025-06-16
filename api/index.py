# query.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from rag_system import generate_rag_answer 

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def query_knowledge_base(request: QueryRequest):
    rag_output = generate_rag_answer(request.question) # Renamed variable for clarity

    # --- START CHANGES HERE ---
    # Ensure rag_output is a dictionary with 'answer' and 'sources' (from rag_system.py)
    # Extract the answer string
    answer_text = rag_output.get("answer", "I'm sorry, I cannot answer this question at this time.")
    
    # Extract sources and format them into 'links' as expected by promptfoo
    # Ensure sources is a list, and each item has 'url' and 'title' (for 'text')
    links_list = []
    if "sources" in rag_output and isinstance(rag_output["sources"], list):
        for source in rag_output["sources"]:
            if "url" in source and "title" in source:
                links_list.append({"url": source["url"], "text": source["title"]})
            elif "url" in source: # Fallback if title is missing
                links_list.append({"url": source["url"], "text": source["url"]})
    # --- END CHANGES HERE ---

    return JSONResponse(content={
        "answer": answer_text, # This is the string answer
        "links": links_list    # This is the list of formatted links
    })

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RAG API is running"}

# Root route for submission system/browser check
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2>Virtual TA Backend is Running ✅</h2>
    <p>Use POST /ask to send questions.<br>
    Or check health at <a href="/health">/health</a>.</p>
    """