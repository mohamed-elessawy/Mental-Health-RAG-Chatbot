from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    intent: str
    emotion: str
    language: str
    response: str
    retrieved_documents: bool
