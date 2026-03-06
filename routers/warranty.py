import uuid

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from service.warranty_agent import process_claim, check_history, validate_claim, health_check

router = APIRouter(prefix="/warranty", tags=["warranty"])

@router.post("/process")
async def process_warranty_claim(
    files: list[UploadFile] = File(...),
    submission_id: str = Form(default=None),
):
    """Forward a warranty claim to the document processing agent."""
    try:
        if submission_id is None:
            submission_id = str(uuid.uuid4())
        f = files[0]
        data = await f.read()
        result = await process_claim(data, f.filename or "unknown", submission_id)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"Warranty processing error: {str(e)}"},
        )


@router.post("/check-history")
async def check_warranty_history(
    failure_description: str = Form(...),
    engine_serial: str | None = Form(default=None),
):
    """Search for similar past claims."""
    try:
        return await check_history(failure_description, engine_serial)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"History check error: {str(e)}"},
        )


@router.post("/validate")
async def validate_warranty_claim(claim_data: dict):
    """Validate extracted claim data."""
    try:
        return await validate_claim(claim_data)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"Validation error: {str(e)}"},
        )


@router.get("/health")
async def warranty_agent_health():
    """Check if the warranty agent is reachable."""
    try:
        return await health_check()
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"Health check error: {str(e)}"},
        )
