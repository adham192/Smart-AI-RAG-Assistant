"""
FastAPI + LangServe backend for Hybrid RAG Assistant
"""

from fastapi import FastAPI, UploadFile, File
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
import shutil
import os
from src.vectorstore import FAISSVectorStore
from src.RAG_pipeline import RAGPipeline
from src.document_loading import DocumentLoader
from src.chunking import DocumentChunker
from src.config import config


# Initialize Components
app = FastAPI(
    title="Smart AI Assistant",
    version="1.0"
)

vectorstore = FAISSVectorStore()
rag_pipeline = RAGPipeline(vectorstore)

loader = DocumentLoader()
chunker = DocumentChunker(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)


# Upload Endpoint
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    document = loader.load_document(file_location)
    chunks = chunker.chunk_document(document)
    vectorstore.add_documents(chunks)

    os.remove(file_location)

    return {
        "message": f"{file.filename} uploaded successfully",
        "total_vectors": vectorstore.get_info()["total_vectors"]
    }


# Hybrid Chat Runnable
async def hybrid_chat(input: str): 
    # Safety guard
    if not input or not input.strip():
        yield "Please provide a valid question."
        return

    for chunk in rag_pipeline.stream_query(input): 
        yield chunk

chat_runnable = RunnableLambda(hybrid_chat)

add_routes(
    app,
    chat_runnable,
    path="/chat"
)

# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
