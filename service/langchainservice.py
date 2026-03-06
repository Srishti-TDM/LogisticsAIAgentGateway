import asyncio
import json
import os
import re

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from models.schema import ChatResponse

AGENT_BASE_URL = os.environ.get(
    "WARRANTY_AGENT_URL",
    "https://read-only-ai-agent-warranty-production.up.railway.app",
)

INTERPRETATION_PROMPT = ChatPromptTemplate.from_template(
    "You are analyzing the response from a logistics {agent_name} agent.\n\n"
    "Agent raw response:\n{raw_response}\n\n"
    "Provide a structured breakdown with these sections:\n"
    "- What was checked\n"
    "- Key findings\n"
    "- Decision / recommendation\n"
    "- Confidence level\n\n"
    "Be concise and use bullet points."
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "You are summarizing the response from a logistics {agent_name} agent.\n\n"
    "Agent raw response:\n{raw_response}\n\n"
    "Write a 1-3 sentence plain English summary suitable for a chat bubble. "
    "No bullet points, no markdown, just clear conversational language."
)


def _extract_part_numbers(message: str) -> list[str]:
    """Pull part numbers from free text (sequences of digits, optionally with dashes)."""
    return re.findall(r"\b\d[\d\-]{3,}\d\b", message)


class LangChainService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    async def run(self, message: str, domain: str = "general") -> ChatResponse:
        if domain == "operations":
            agent_used = "operations"
            raw = await self._call_operations_agent(message)
        elif domain == "risk":
            agent_used = "risk_management"
            raw = await self._call_risk_agent(message)
        else:
            agent_used = "general"
            raw = await self._call_risk_agent(message)

        if "error" in raw:
            return ChatResponse(
                agent_used=agent_used,
                agent_raw_response=raw,
                gateway_interpretation=raw["error"],
                simplified_summary=raw["error"],
            )

        raw_str = json.dumps(raw, indent=2, default=str)

        interpretation_chain = INTERPRETATION_PROMPT | self.llm
        summary_chain = SUMMARY_PROMPT | self.llm

        interp_result, summary_result = await asyncio.gather(
            interpretation_chain.ainvoke({"agent_name": agent_used, "raw_response": raw_str}),
            summary_chain.ainvoke({"agent_name": agent_used, "raw_response": raw_str}),
        )

        return ChatResponse(
            agent_used=agent_used,
            agent_raw_response=raw,
            gateway_interpretation=interp_result.content,
            simplified_summary=summary_result.content,
        )

    async def _call_risk_agent(self, message: str) -> dict:
        """Call the warranty claim validation agent with the user's original message."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{AGENT_BASE_URL}/claims/validate",
                    json={
                        "failure_description": message,
                        "claim_id": "",
                        "engine_serial": "",
                        "part_numbers": [],
                        "failure_date": None,
                        "service_date": None,
                        "dealer_code": "",
                        "technician_id": "",
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as e:
            return {"error": f"Risk agent unreachable: {str(e)}"}

    async def _call_operations_agent(self, message: str) -> dict:
        """Call the inventory agent. Uses check-availability if part numbers are found,
        otherwise falls back to catalog search."""
        part_numbers = _extract_part_numbers(message)
        if part_numbers:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{AGENT_BASE_URL}/inventory/check-availability",
                        json={"part_numbers": part_numbers},
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPError as e:
                return {"error": f"Operations agent unreachable: {str(e)}"}
        else:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.get(
                        f"{AGENT_BASE_URL}/inventory/search",
                        params={"q": message},
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPError as e:
                return {"error": f"Operations agent unreachable: {str(e)}"}
