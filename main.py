from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from routers import chat, warranty
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(chat.router)
app.include_router(warranty.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)