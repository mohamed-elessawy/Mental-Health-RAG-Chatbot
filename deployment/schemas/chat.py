from typing import List, Optional

from pydantic import BaseModel


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
