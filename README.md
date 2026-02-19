# Smart AI/RAG Assistant
A production-ready Retrieval-Augmented Generation (RAG) system that lets you upload documents (PDF/DOCX) and chat with them, powered by Google Gemini, 
FAISS vector search, LangChain, and a Gradio frontend served through a FastAPI + LangServe backend with real-time streaming responses.

## 🚀 Features
* 📄 Upload **PDF** and **DOCX** documents via a browser UI
* 🔍 Semantic search powered by **FAISS** and **Google Gemini embeddings**
* 🤖 Conversational AI backed by **Gemini 2.5 Flash**
* ⚡ **Real-time streaming** responses (token by token)
* 🧠 **Hybrid intelligence:** uses document context when relevant, falls back to general knowledge when not
* 🔧 Clean modular codebase — each component is independently replaceable

## 🏗 Project Architecture

The chatbot has been refactored into a clean, modular architecture:
```
smart_rag_assistant/
│
├── frontend.py                 # Gradio UI — file upload + streaming chat interface
├── server.py                   # FastAPI + LangServe backend entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
│
└── src/                        # Core application modules
    ├── config.py               # Centralized settings (models, API keys, chunk sizes)
    ├── document_loading.py     # PDF + DOCX text extraction and parsing
    ├── chunking.py             # Recursive text splitting into overlapping chunks
    ├── embedding.py            # Google Gemini embedding API wrapper
    ├── vectorstore.py          # FAISS index management and similarity search
    └── RAG_pipeline.py         # RAG orchestration — retrieval, prompting, streaming
```

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/adham192/Smart-AI-RAG-Assistant.git
cd Smart-AI-RAG-Assistant
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables:**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 🎯 Usage

**Start the application:**
1. **Run the backend server:**
```bash
python server.py
# Starts at http://127.0.0.1:8000
```

2. **Run the frontend UI** *(in a separate terminal):*
```bash
python frontend.py
# Opens at http://127.0.0.1:7860
```

3. **Upload a Document:**
   - Use the sidebar to upload PDF or DOCX files

4. **Index Documents:**
   - Click "Upload & Index" to process and index the documents
   - The system will create embeddings and build a vector database

5. **Ask Questions:**
   - Ask any question related or not to the document

## 🏛️ Module Overview

### Entry Points
#### `frontend.py`
- Gradio-based browser UI
- Document upload panel with status feedback
- Streaming chat interface with conversation history
- Communicates with the backend via HTTP requests

#### `server.py`
- FastAPI + LangServe backend entry point
- `/upload` endpoint for document ingestion
- `/chat/stream` endpoint for real-time streaming responses
- Wires all core modules together



### Core Modules (`src/`)
#### `config.py`
- Centralized configuration management
- Loads API keys from `.env` file
- Model names, chunking settings, retrieval parameters

#### `document_loading.py`
- Text extraction from PDF files using PyMuPDF
- Text extraction from DOCX files using python-docx
- Returns standardized document dictionaries with metadata

#### `chunking.py`
- Recursive character text splitting
- Configurable chunk size and overlap
- Preserves natural text boundaries (paragraphs, sentences, words)

#### `embedding.py`
- Google Gemini embedding API wrapper
- Separate modes for query vs. document embedding
- Auto-detects embedding dimension on initialization

#### `vectorstore.py`
- FAISS index creation and management
- Adds and stores document chunk embeddings
- Similarity search with and without distance scores
- Save/load index to and from disk

#### `RAG_pipeline.py`
- Core RAG orchestration logic
- Smart fallback: uses document context when relevant, general knowledge otherwise
- Real-time token streaming via LangChain + Gemini 2.5 Flash

## 🔮 Future Work

- **Public Deployment** — Host the application on a cloud platform (e.g., Hugging Face Spaces) to make it publicly accessible without local setup
- **Additional Document Formats** — Extend support beyond PDF and DOCX to include TXT, CSV, PPTX, Excel, and web URLs
- **Multi-Document Management** — Allow users to view, delete, or switch between individually indexed documents from the UI
- **Authentication & User Sessions** — Add login support so each user maintains their own private document index and chat history
- **Support for Larger Documents** — Implement smarter chunking strategies and batch embedding to handle very large files without timeouts
- **Multilingual Support** — Enable document ingestion and querying in languages other than English
