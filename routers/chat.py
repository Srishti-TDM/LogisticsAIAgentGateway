from fastapi import APIRouter
from fastapi.responses import JSONResponse
from service.langchainservice import LangChainService
from models.schema import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])
service = LangChainService()

@router.post("/")
async def chat_endpoint(request: ChatRequest):
    try:
        result = await service.run(request.message, request.domain)
        return {"response": result}
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"Chat service error: {str(e)}"},
        )