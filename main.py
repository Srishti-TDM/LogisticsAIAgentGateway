from fastapi import FastAPI
from routers import chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(chat.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whats left: langchain service.py result variable should be able to delegate to MCP server