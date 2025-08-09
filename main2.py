# --- Memory-Optimized Python RAG Q&A Service with FastAPI ---
# Optimized for cloud deployment with limited memory (512MB)

import os
import json
import gc
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import uvicorn
import uuid
import requests
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Qdrant specific imports
from qdrant_client import QdrantClient, models

# RAG Pipeline imports
import fitz
import tiktoken
import math
import re
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Environment Variables ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", 8001))  # Use PORT from environment for Render

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG Q&A Service", version="1.0.0")

# --- Global Services (lazy loading to save memory) ---
embedding_model = None
qdrant_client = None

# --- RAG Pipeline Configuration (reduced for memory efficiency) ---
CHUNK_SIZE_TOKENS = 256  # Reduced from 512
OVERLAP_PERCENTAGE = 0.1  # Reduced from 0.15
ENCODING_NAME = "cl100k_base"
MAX_LINES_TO_CHECK = 3  # Reduced from 5
REPETITION_THRESHOLD_PERCENT = 70

ENCODER = tiktoken.get_encoding(ENCODING_NAME)
COLLECTION_NAME = "docs"

# --- Helper Functions ---

def call_gemini_api(prompt: str) -> str:
    """Calls the Gemini API to get a text response."""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 512,  # Limit response length
            "temperature": 0.1
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        
        return response_json['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error generating response."

def get_embedding_model():
    """Lazy load embedding model to save memory."""
    global embedding_model
    if embedding_model is None:
        try:
            # Use a smaller, more efficient model
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise HTTPException(status_code=503, detail="Embedding model unavailable")
    return embedding_model

def setup_qdrant():
    """Setup Qdrant client."""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(":memory:")
        
        # Create collection with smaller vector size
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        print("✅ Qdrant initialized")
    else:
        # Recreate collection for fresh state
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
        except:
            pass
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )

def count_tokens(text: str) -> int:
    """Counts tokens using the global tiktoken encoder."""
    return len(ENCODER.encode(text))

def download_pdf(url: str) -> bytes:
    """Downloads PDF from URL with memory optimization."""
    try:
        # Stream download to avoid loading entire file in memory
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Read in chunks to manage memory
        pdf_data = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            pdf_data.extend(chunk)
        
        return bytes(pdf_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF download failed: {e}")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict]:
    """Extracts text from PDF with memory optimization."""
    pages_content = []
    document = None
    
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Limit pages to avoid memory issues
        max_pages = min(len(document), 50)  # Process max 50 pages
        
        for page_num in range(max_pages):
            page = document.load_page(page_num)
            text = page.get_text("text")
            
            # Only keep pages with substantial content
            if len(text.strip()) > 100:
                pages_content.append({"page_num": page_num + 1, "text": text})
            
            # Clean up page from memory
            del page
            
    except Exception as e:
        print(f"PDF extraction error: {e}")
    finally:
        if document:
            document.close()
        
        # Force garbage collection
        gc.collect()
    
    return pages_content

def simple_chunk_text(text: str, doc_id: str, page: int) -> List[Dict]:
    """Simplified chunking to save memory."""
    # Simple sentence-based chunking
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if count_tokens(test_chunk) > CHUNK_SIZE_TOKENS and current_chunk:
            # Save current chunk
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    "doc_id": doc_id,
                    "page": page,
                    "chunk_id": f"{doc_id}-p{page}-c{len(chunks)}"
                },
                "point_id": str(uuid.uuid4())
            })
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "metadata": {
                "doc_id": doc_id,
                "page": page,
                "chunk_id": f"{doc_id}-p{page}-c{len(chunks)}"
            },
            "point_id": str(uuid.uuid4())
        })
    
    return chunks

def process_pdf_memory_efficient(pdf_bytes: bytes, doc_id: str) -> List[Dict]:
    """Memory-efficient PDF processing."""
    print("📄 Extracting text...")
    pages_content = extract_text_from_pdf_bytes(pdf_bytes)
    
    if not pages_content:
        raise ValueError("No content extracted from PDF")
    
    print(f"📄 Processing {len(pages_content)} pages...")
    all_chunks = []
    
    for page_data in pages_content:
        # Simple text cleaning
        text = page_data['text']
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) > 50:  # Only process substantial content
            page_chunks = simple_chunk_text(text, doc_id, page_data['page_num'])
            all_chunks.extend(page_chunks)
    
    # Clean up
    del pages_content
    gc.collect()
    
    return all_chunks

def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """Retrieve relevant chunks."""
    try:
        model = get_embedding_model()
        query_vector = model.encode(query).tolist()
        
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        chunks = []
        for result in search_results:
            chunks.append({
                'content': result.payload.get('content', ''),
                'metadata': {k: v for k, v in result.payload.items() if k != 'content'},
                'score': result.score
            })
        return chunks
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def generate_answer(question: str, chunks: List[Dict]) -> str:
    """Generate answer with limited context."""
    if not GEMINI_API_KEY:
        return "API key not configured"
    
    # Limit context to avoid token limits
    context = "\n\n".join([chunk['content'][:300] for chunk in chunks[:3]])
    
    prompt = f"""Based on the following context, answer the question concisely:

Context:
{context}

Question: {question}

Answer (be specific and concise):"""
    
    return call_gemini_api(prompt)

# --- Request/Response Models ---

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --- Main Endpoint ---

@app.post("/qa", response_model=QAResponse)
async def question_answering(request: QARequest):
    """Memory-optimized Q&A endpoint."""
    try:
        print(f"🔍 Processing {len(request.questions)} questions")
        
        # Initialize services
        setup_qdrant()
        model = get_embedding_model()
        
        # Process document
        print("📥 Downloading PDF...")
        pdf_bytes = download_pdf(request.documents)
        
        print("📄 Processing PDF...")
        doc_id = str(uuid.uuid4())[:8]  # Shorter IDs
        chunks = process_pdf_memory_efficient(pdf_bytes, doc_id)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted")
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Index chunks in batches to manage memory
        print("🔍 Indexing chunks...")
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            points = []
            
            for chunk in batch:
                try:
                    vector = model.encode(chunk['content']).tolist()
                    points.append(
                        models.PointStruct(
                            id=chunk['point_id'],
                            vector=vector,
                            payload=chunk['metadata'] | {"content": chunk['content']}
                        )
                    )
                except Exception as e:
                    print(f"Encoding error: {e}")
                    continue
            
            if points:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            
            # Clean up batch from memory
            del batch, points
            gc.collect()
        
        print("✅ Indexing complete")
        
        # Process questions
        answers = []
        for i, question in enumerate(request.questions):
            print(f"❓ Question {i+1}/{len(request.questions)}")
            
            relevant_chunks = retrieve_chunks(question, top_k=5)
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer.strip())
            
            # Clean up
            del relevant_chunks
            gc.collect()
        
        # Final cleanup
        del chunks, pdf_bytes
        gc.collect()
        
        print("✅ Complete!")
        return QAResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        # Clean up on error
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memory_usage": "optimized",
        "services": {
            "gemini": GEMINI_API_KEY is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "🚀 Memory-Optimized RAG Q&A Service",
        "status": "running",
        "endpoint": "POST /qa"
    }

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
