from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    domain: str = "general"

