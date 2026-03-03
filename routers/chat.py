from fastapi import APIRouter
from service.langchainservice import LangChainService
from models.schema import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])
service = LangChainService()

@router.post("/")
async def chat_endpoint(request: ChatRequest):
    result = await service.run(request.message, request.domain)
    return {"response": result}