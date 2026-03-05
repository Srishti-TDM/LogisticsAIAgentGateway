import os
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

RISK_AGENT_URL = os.environ.get(
    "WARRANTY_AGENT_URL",
    "https://read-only-ai-agent-warranty-production.up.railway.app",
)

RISK_PROMPT = ChatPromptTemplate.from_template(
    "You are a logistics risk management assistant. "
    "Analyze the following request and provide a risk assessment.\n\n"
    "User request: {message}"
)


class LangChainService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    async def run(self, message: str, domain: str = "risk"):
        chain = RISK_PROMPT | self.llm
        response = await chain.ainvoke({"message": message})
        lc_response = response.content

        agent_response = await self._call_risk_agent(lc_response)

        return {
            "langchain_response": lc_response,
            "agent_response": agent_response,
            "domain": domain,
        }

    async def _call_risk_agent(self, message: str) -> dict:
        """Call the warranty claim processing agent with extracted text."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{RISK_AGENT_URL}/claims/validate",
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
