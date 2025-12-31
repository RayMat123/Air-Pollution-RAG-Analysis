from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import generate_answer

app = FastAPI(title="Air Pollution Chatbot")

# Health check
@app.get("/")
def health_check():
    return {"status": "Chatbot running"}

# Request body model for /chat
class ChatQuery(BaseModel):
    question: str

# Chat endpoint
@app.post("/chat")
def chat(query: ChatQuery):
    try:
        answer = generate_answer(query.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
