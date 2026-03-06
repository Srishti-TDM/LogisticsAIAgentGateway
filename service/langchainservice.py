import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

AGENT_BASE_URL = "https://read-only-ai-agent-warranty-production.up.railway.app"

RISK_PROMPT = ChatPromptTemplate.from_template(
    "You are a logistics risk management assistant. "
    "Analyze the following request and provide a risk assessment.\n\n"
    "User request: {message}"
)

OPERATIONS_PROMPT = ChatPromptTemplate.from_template(
    "You are a logistics operations assistant. "
    "Analyze the following request and provide operational guidance "
    "including parts availability, inventory status, and supply chain recommendations.\n\n"
    "User request: {message}"
)

GENERAL_PROMPT = ChatPromptTemplate.from_template(
    "You are a logistics assistant. "
    "Analyze the following request and provide helpful guidance.\n\n"
    "User request: {message}"
)

DOMAIN_PROMPTS = {
    "risk": RISK_PROMPT,
    "operations": OPERATIONS_PROMPT,
    "general": GENERAL_PROMPT,
}


class LangChainService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    async def run(self, message: str, domain: str = "general"):
        prompt = DOMAIN_PROMPTS.get(domain, GENERAL_PROMPT)
        chain = prompt | self.llm
        response = await chain.ainvoke({"message": message})
        lc_response = response.content

        if domain == "risk":
            agent_response = await self._call_risk_agent(lc_response)
        elif domain == "operations":
            agent_response = await self._call_operations_agent(lc_response)
        else:
            agent_response = None

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
        """Call the inventory/operations agent.

        Uses the search endpoint for general queries and the
        check-availability endpoint when part numbers are provided.
        """
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
