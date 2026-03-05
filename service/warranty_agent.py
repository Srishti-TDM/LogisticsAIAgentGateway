import os
import httpx
import os

WARRANTY_AGENT_URL = os.environ.get(
    "WARRANTY_AGENT_URL",
    "https://read-only-ai-agent-warranty-production.up.railway.app",
)


async def process_claim(file_bytes: bytes, filename: str, submission_id: str) -> dict:
    """Send a file to the warranty claim processing agent."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{WARRANTY_AGENT_URL}/claims/process",
                files={"files": (filename, file_bytes)},
                data={"submission_id": submission_id},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        return {"error": f"Warranty agent error: {str(e)}"}


async def check_history(failure_description: str, engine_serial: str | None = None) -> dict:
    """Search service history for similar claims."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            data = {"failure_description": failure_description}
            if engine_serial:
                data["engine_serial"] = engine_serial
            resp = await client.post(
                f"{WARRANTY_AGENT_URL}/claims/check-history",
                data=data,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        return {"error": f"History check error: {str(e)}"}


async def validate_claim(claim_data: dict) -> dict:
    """Validate extracted claim data."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{WARRANTY_AGENT_URL}/claims/validate",
                json=claim_data,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        return {"error": f"Validation error: {str(e)}"}


async def health_check() -> dict:
    """Check if the warranty agent is online."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{WARRANTY_AGENT_URL}/health")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        return {"error": f"Health check error: {str(e)}"}
