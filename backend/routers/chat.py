"""Chat API endpoints."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # 1 levels to root

from fastapi import APIRouter, HTTPException
from schemas.chat import ChatMessageRequest, ChatMessageResponse
from core.dependencies import LlmDep, DatabaseDep

router = APIRouter()


@router.post("/", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest, llm: LlmDep, db: DatabaseDep
):
    """Send a chat message and get AI response."""
    # TODO: Implement chat logic with RAG
    # 1. Create or get session
    # 2. Save user message to database
    # 3. Retrieve relevant context from vector database
    # 4. Generate AI response with OpenAI
    # 5. Save assistant message to database
    # 6. Return response with RAGAS scores

    raise HTTPException(
        status_code=501, detail="Chat functionality not implemented yet"
    )


@router.get("/sessions")
async def get_sessions(db: DatabaseDep):
    """Get user's chat sessions."""
    # TODO: Implement session listing
    return {"sessions": []}


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, db: DatabaseDep):
    """Get chat history for a session."""
    # TODO: Implement chat history retrieval
    return {"session_id": session_id, "messages": []}
