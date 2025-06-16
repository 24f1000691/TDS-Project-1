from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
import traceback

try:
    from rag_system import generate_rag_answer
except Exception as e:
    print("❌ Failed to import rag_system:")
    traceback.print_exc()
    generate_rag_answer = lambda q: {"answer": "Error loading RAG system", "sources": []}

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def query_knowledge_base(request: QueryRequest):
    try:
        rag_output = generate_rag_answer(request.question)
        answer_text = rag_output.get("answer", "I'm sorry, I cannot answer this question at this time.")
        links_list = []
        if "sources" in rag_output and isinstance(rag_output["sources"], list):
            for source in rag_output["sources"]:
                if "url" in source and "title" in source:
                    links_list.append({"url": source["url"], "text": source["title"]})
                elif "url" in source:
                    links_list.append({"url": source["url"], "text": source["url"]})
        return JSONResponse(content={"answer": answer_text, "links": links_list})
    except Exception as e:
        print("❌ Error inside /ask route:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.get("/")
async def root():
    return HTMLResponse("""
        <h2>Virtual TA Backend is Running ✅</h2>
        <p>Use POST /ask to send questions.<br>
        Or check health at /health.</p>
    """)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RAG API is running"}
