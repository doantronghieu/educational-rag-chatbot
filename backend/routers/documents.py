"""Document management API endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from ..schemas.document import DocumentUploadResponse, DocumentListResponse
from ..core.dependencies import StorageDep, VectorStoreDep, DatabaseDep

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    storage_client: StorageDep = None,
    vector_store_client: VectorStoreDep = None,
    db: DatabaseDep = None,
):
    """Upload a PDF document for knowledge base."""
    # TODO: Implement document upload and processing
    # 1. Validate file type and size
    # 2. Upload to MinIO storage
    # 3. Extract text using Unstructured.IO
    # 4. Chunk text with LangChain
    # 5. Generate embeddings with OpenAI
    # 6. Store chunks in Qdrant
    # 7. Save metadata to database

    raise HTTPException(status_code=501, detail="Document upload not implemented yet")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(db: DatabaseDep, page: int = 1, limit: int = 20):
    """List uploaded documents."""
    # TODO: Implement document listing with pagination
    return {"documents": [], "total": 0}


@router.delete("/{document_id}")
async def delete_document(
    document_id: str, db: DatabaseDep, storage_client: StorageDep, vector_store_client: VectorStoreDep
):
    """Delete a document and its chunks."""
    # TODO: Implement document deletion
    # 1. Remove from Qdrant vector database
    # 2. Remove from MinIO storage
    # 3. Remove from database

    raise HTTPException(status_code=501, detail="Document deletion not implemented yet")
