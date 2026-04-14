"""
LLM Learning Platform - FastAPI Backend
"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import config  # ensures sys.path is set up
import database

from routers import tokenizer, architecture, attention, training, chat
from websockets.training_ws import training_websocket

app = FastAPI(
    title="LLM Learning Platform API",
    description="Interactive LLM learning platform based on 'Build a Large Language Model From Scratch'",
    version="1.0.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
async def startup():
    database.init_db()


# Include routers
app.include_router(tokenizer.router)
app.include_router(architecture.router)
app.include_router(attention.router)
app.include_router(training.router)
app.include_router(chat.router)


# WebSocket endpoints
@app.websocket("/ws/training/{run_id}")
async def ws_training(websocket: WebSocket, run_id: str):
    await training_websocket(websocket, run_id)


@app.get("/")
def root():
    return {
        "name": "LLM Learning Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "modules": ["tokenizer", "architecture", "attention", "training", "chat"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
