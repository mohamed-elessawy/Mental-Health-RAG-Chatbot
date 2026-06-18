from typing import Optional

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str
    content: str
    language: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class ChatResponse(BaseModel):
    intent: str
    emotion: str
    language: str
    response: str
    retrieved_documents: bool


class FeedbackRequest(BaseModel):
    vote: str
    user_message: str
    bot_response: str

    @field_validator("vote")
    @classmethod
    def vote_must_be_valid(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError("vote must be 'up' or 'down'")
        return v


class FeedbackResponse(BaseModel):
    status: str
