import httpx
import os

WARRANTY_API_KEY = os.getenv("WARRANTY_AGENT_URL")

async def process_claim(file_bytes: bytes, filename: str, submission_id: str) -> dict:
    """Send a file to the warranty claim processing agent."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{WARRANTY_API_KEY}/claims/process",
            files={"files": (filename, file_bytes)},
            data={"submission_id": submission_id},
        )
        resp.raise_for_status()
        return resp.json()


async def check_history(failure_description: str, engine_serial: str | None = None) -> dict:
    """Search service history for similar claims."""
    async with httpx.AsyncClient(timeout=30) as client:
        data = {"failure_description": failure_description}
        if engine_serial:
            data["engine_serial"] = engine_serial
        resp = await client.post(
            f"{WARRANTY_API_KEY}/claims/check-history",
            data=data,
        )
        resp.raise_for_status()
        return resp.json()


async def validate_claim(claim_data: dict) -> dict:
    """Validate extracted claim data."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{WARRANTY_API_KEY}/claims/validate",
            json=claim_data,
        )
        resp.raise_for_status()
        return resp.json()


async def health_check() -> dict:
    """Check if the warranty agent is online."""
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(f"{WARRANTY_API_KEY}/health")
        resp.raise_for_status()
        return resp.json()
