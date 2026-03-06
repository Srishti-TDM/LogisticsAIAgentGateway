from fastapi import APIRouter
from fastapi.responses import JSONResponse
from service.langchainservice import LangChainService
from models.schema import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])
service = LangChainService()


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        return await service.run(request.message, request.domain)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"Chat service error: {str(e)}"},
        )
