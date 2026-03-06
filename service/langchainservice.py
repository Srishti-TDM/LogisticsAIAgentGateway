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

CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    "You are a logistics request classifier. "
    "Determine whether the following user request is about:\n"
    "- 'risk': warranty claims, failures, risk assessments, engine problems, damage reports\n"
    "- 'operations': inventory, parts availability, stock levels, warehouses, supply chain\n\n"
    "Respond with ONLY the single word 'risk' or 'operations'. Nothing else.\n\n"
    "User request: {message}"
)

SEARCH_KEYWORD_PROMPT = ChatPromptTemplate.from_template(
    "Extract a short search keyword (1-3 words) from this user request that would match "
    "inventory part descriptions in a database. Use the actual component name that would "
    "appear in a parts catalog (e.g. 'turbocharger' not 'turbo', 'water pump' not 'pump', "
    "'engine oil' not 'lubrication').\n\n"
    "Also determine if it matches one of these categories: "
    "lubricants, filtration, cooling, fuel_system, exhaust.\n\n"
    "User request: {message}\n\n"
    "Respond in EXACTLY this format (no other text):\n"
    "keyword: <search term>\n"
    "category: <category or none>"
)


_DOMAIN_REQUEST_RE = re.compile(
    r"^Domain:\s*(?P<domain>\S+)\s+User Request:\s*(?P<request>.+)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_message(raw_message: str) -> tuple[str, str]:
    """Parse 'Domain: {domain} User Request: {request}' format.

    Returns (domain, user_request). Falls back to ("general", raw_message)
    if the format doesn't match.
    """
    m = _DOMAIN_REQUEST_RE.match(raw_message.strip())
    if m:
        return m.group("domain").lower(), m.group("request").strip()
    return "general", raw_message


def _extract_part_numbers(message: str) -> list[str]:
    """Pull part numbers from free text (sequences of digits, optionally with dashes)."""
    return re.findall(r"\b\d[\d\-]{3,}\d\b", message)


class LangChainService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    async def run(self, message: str, domain: str = "general") -> ChatResponse:
        # Parse "Domain: X User Request: Y" format from the UI
        parsed_domain, user_request = _parse_message(message)
        # Prefer the domain extracted from the message; fall back to the field
        domain = parsed_domain if parsed_domain != "general" else domain

        if domain == "operations":
            agent_used = "operations"
            raw = await self._call_operations_agent(user_request)
        elif domain == "risk":
            agent_used = "risk_management"
            raw = await self._call_risk_agent(user_request)
        else:
            # Classify the request and route to the right agent
            domain = await self._classify(user_request)
            if domain == "operations":
                agent_used = "operations"
                raw = await self._call_operations_agent(user_request)
            else:
                agent_used = "risk_management"
                raw = await self._call_risk_agent(user_request)

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

    async def _classify(self, message: str) -> str:
        """Use the LLM to classify a request as 'risk' or 'operations'."""
        chain = CLASSIFIER_PROMPT | self.llm
        response = await chain.ainvoke({"message": message})
        result = response.content.strip().lower()
        return result if result in ("risk", "operations") else "risk"

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

    async def _extract_search_params(self, message: str) -> tuple[str, str | None]:
        """Use LLM to extract a search keyword and optional category from the user message."""
        chain = SEARCH_KEYWORD_PROMPT | self.llm
        response = await chain.ainvoke({"message": message})
        text = response.content.strip()

        keyword = message  # fallback
        category = None
        for line in text.splitlines():
            line = line.strip().lower()
            if line.startswith("keyword:"):
                keyword = line.split(":", 1)[1].strip()
            elif line.startswith("category:"):
                val = line.split(":", 1)[1].strip()
                if val and val != "none":
                    category = val
        return keyword, category

    async def _call_operations_agent(self, message: str) -> dict:
        """Call the inventory agent. Uses check-availability if part numbers are found,
        otherwise extracts search keywords and queries the catalog."""
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
            keyword, category = await self._extract_search_params(message)
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    # Try 1: keyword + category
                    params: dict[str, str | int] = {"q": keyword}
                    if category:
                        params["category"] = category
                    resp = await client.get(
                        f"{AGENT_BASE_URL}/inventory/search", params=params,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    # Try 2: keyword only (no category filter)
                    if result.get("count", 0) == 0 and category:
                        resp = await client.get(
                            f"{AGENT_BASE_URL}/inventory/search", params={"q": keyword},
                        )
                        resp.raise_for_status()
                        result = resp.json()

                    # Try 3: all parts in the category
                    if result.get("count", 0) == 0 and category:
                        resp = await client.get(
                            f"{AGENT_BASE_URL}/inventory/search",
                            params={"q": "%", "category": category},
                        )
                        resp.raise_for_status()
                        result = resp.json()

                    return result
            except httpx.HTTPError as e:
                return {"error": f"Operations agent unreachable: {str(e)}"}
