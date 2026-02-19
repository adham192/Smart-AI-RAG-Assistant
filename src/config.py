"""
Configuration file for Smart Contract Assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Settings
    LLM_MODEL = "models/gemini-2.5-flash"
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    TEMPERATURE = 0.5
    MAX_OUTPUT_TOKENS = 2048
    
    
    # Chunking Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Settings
    TOP_K_RETRIEVAL = 4
    SIMILARITY_THRESHOLD = 50.0
    
    # Server Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # UI Settings
    GRADIO_SHARE = True
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS = [".pdf", ".docx"]
    
    # Response Settings
    MAX_CHAT_HISTORY = 10
    INCLUDE_SOURCES = True
    
    # Evaluation Settings
    EVAL_METRICS = ["relevance", "faithfulness", "answer_quality"]

# Create instance
config = Config()
