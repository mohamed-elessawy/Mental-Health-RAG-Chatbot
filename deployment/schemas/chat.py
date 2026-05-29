from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str
    language: Optional[str] = None

class ChatRequest(BaseModel):
    text: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    intent: str
    emotion: str
    language: str
    response: str
    retrieved_documents: bool