from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    domain: str = "general"


class ChatResponse(BaseModel):
    agent_used: str  # "risk_management" | "operations"
    agent_raw_response: dict[str, Any]
    gateway_interpretation: str
    simplified_summary: str

