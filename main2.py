# --- Python RAG Q&A Service with FastAPI ---
# This service takes documents and questions as input and returns answers in JSON format.

# Requirements:
# pip install fastapi uvicorn "python-dotenv[extra]" PyMuPDF tiktoken langchain-text-splitters qdrant-client sentence-transformers requests

import os
import json
import io
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
from sentence_transformers import SentenceTransformer

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

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG Q&A Service", version="1.0.0")

# --- Global Services ---
qdrant_client = None
embedding_model = None

# --- RAG Pipeline Configuration ---
CHUNK_SIZE_TOKENS = 512
OVERLAP_PERCENTAGE = 0.15
ENCODING_NAME = "cl100k_base"
MAX_LINES_TO_CHECK = 5
REPETITION_THRESHOLD_PERCENT = 70

ENCODER = tiktoken.get_encoding(ENCODING_NAME)
COLLECTION_NAME = "policy-documents"

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
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        return response_json['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "An error occurred while communicating with the LLM."
    except KeyError:
        print("Invalid response from Gemini API.")
        return "An error occurred while parsing the LLM response."

def count_tokens(text: str) -> int:
    """Counts tokens using the global tiktoken encoder."""
    return len(ENCODER.encode(text))

def download_pdf(url: str) -> bytes:
    """Downloads PDF from URL and returns bytes."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict]:
    """Extracts text content page by page from PDF bytes (in-memory)."""
    pages_content = []
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text("text")
            pages_content.append({"page_num": page_num + 1, "text": text})
        document.close()
    except Exception as e:
        print(f"Error reading PDF from bytes: {e}")
    return pages_content

def identify_common_page_elements(all_pages_content: dict[str, list[dict]],
                                   max_lines: int = MAX_LINES_TO_CHECK,
                                   repetition_threshold_percent: int = REPETITION_THRESHOLD_PERCENT) -> tuple[set, set]:
    """Analyzes text from multiple pages to identify common header and footer lines."""
    header_candidates = Counter()
    footer_candidates = Counter()
    total_non_first_pages = 0
    
    for doc_id, pages_data in all_pages_content.items():
        for page_data in pages_data:
            page_num = page_data['page_num']
            page_text = page_data['text']
            if page_num == 1: 
                continue
            total_non_first_pages += 1
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            
            for i in range(min(max_lines, len(lines))):
                header_candidates[lines[i]] += 1
                
            for i in range(max(0, len(lines) - max_lines), len(lines)):
                footer_candidates[lines[i]] += 1
    
    common_header_lines = set()
    common_footer_lines = set()
    
    if total_non_first_pages == 0: 
        return common_header_lines, common_footer_lines
    
    threshold_count = math.ceil(total_non_first_pages * (repetition_threshold_percent / 100))
    
    for line, count in header_candidates.items():
        if count >= threshold_count: 
            common_header_lines.add(line)
            
    for line, count in footer_candidates.items():
        if count >= threshold_count and (re.fullmatch(r'\s*\d+\s*', line) or 
                                        re.fullmatch(r'Page\s+\d+\s*(of\s+\d+)?', line, re.IGNORECASE)):
            common_footer_lines.add(line)
    
    return common_header_lines, common_footer_lines

def remove_identified_elements(page_text: str, page_num: int,
                               common_header_lines: set, common_footer_lines: set) -> str:
    """Removes identified common header and footer lines from a page's text."""
    if page_num == 1: 
        return page_text
        
    lines = [line.strip() for line in page_text.split('\n')]
    final_lines = []
    temp_lines = []
    
    for i, line in enumerate(lines):
        if line in common_header_lines and i < MAX_LINES_TO_CHECK: 
            continue
        else: 
            temp_lines.append(line)
    
    footer_check_start_index = max(0, len(temp_lines) - MAX_LINES_TO_CHECK)
    for i, line in enumerate(temp_lines):
        if line in common_footer_lines and i >= footer_check_start_index: 
            continue
        else: 
            final_lines.append(line)
    
    return "\n".join(line for line in final_lines if line.strip() != "")

def chunk_text_with_metadata(text: str, chunk_size_tokens: int, overlap_percentage: float,
                             doc_id: str, page: int, base_clause_id_prefix: str):
    """Splits a given text into chunks using RecursiveCharacterTextSplitter and adds metadata."""
    avg_chars_per_token = 4
    chunk_size_chars = chunk_size_tokens * avg_chars_per_token
    overlap_chars = math.floor(chunk_size_chars * overlap_percentage)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    raw_chunks = text_splitter.split_text(text)
    processed_chunks = []
    
    for i, chunk_content in enumerate(raw_chunks):
        token_length = count_tokens(chunk_content)
        clause_id = f"{base_clause_id_prefix}-{doc_id}-p{page}-c{i + 1}"
        metadata = {
            "doc_id": doc_id,
            "page": page,
            "clause_id": clause_id,
            "chunk_length_tokens": token_length,
            "chunk_length_chars": len(chunk_content)
        }
        processed_chunks.append({
            "content": chunk_content,
            "metadata": metadata,
            "point_id": str(uuid.uuid4())
        })
    
    return processed_chunks

def process_single_pdf_bytes(pdf_bytes: bytes, doc_id: str) -> List[Dict]:
    """Processes a single PDF from bytes, cleans it, and chunks the content."""
    pages_content = extract_text_from_pdf_bytes(pdf_bytes)
    if not pages_content:
        raise ValueError("Failed to extract content from PDF.")
        
    all_docs_pages_content = {doc_id: pages_content}
    common_header_lines, common_footer_lines = identify_common_page_elements(all_docs_pages_content)
    
    all_processed_chunks = []
    for page_data in pages_content:
        cleaned_page_text = remove_identified_elements(
            page_data['text'], page_data['page_num'], common_header_lines, common_footer_lines
        )
        if cleaned_page_text.strip():
            page_chunks = chunk_text_with_metadata(
                text=cleaned_page_text,
                chunk_size_tokens=CHUNK_SIZE_TOKENS,
                overlap_percentage=OVERLAP_PERCENTAGE,
                doc_id=doc_id,
                page=page_data['page_num'],
                base_clause_id_prefix="Clause"
            )
            all_processed_chunks.extend(page_chunks)
            
    return all_processed_chunks

def retrieve_relevant_chunks(query: str, qdrant_client: QdrantClient, embedding_model: SentenceTransformer, top_k: int = 5) -> List[Dict]:
    """Retrieves relevant chunks from Qdrant based on semantic similarity."""
    try:
        query_vector = embedding_model.encode(query).tolist()
        
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        chunks = []
        for result in search_results:
            chunk_data = {
                'content': result.payload.get('content', ''),
                'metadata': {k: v for k, v in result.payload.items() if k != 'content'},
                'score': result.score
            }
            chunks.append(chunk_data)
        return chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []

def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """Generates an answer using the retrieved context with Gemini API."""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."
    
    try:
        # Prepare context
        context_text = "\n\n".join([chunk['content'] for chunk in context_chunks[:5]])
        
        # Create prompt for answer generation
        prompt = f"""
You are an expert assistant that provides accurate and concise answers based on the provided context.

Context:
{context_text}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context
2. If the context doesn't contain enough information to answer the question, state that clearly
3. Be concise but complete in your answer
4. Use specific details from the context when possible
5. Do not make assumptions or add information not present in the context

Answer:"""

        return call_gemini_api(prompt)
        
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return f"Error generating answer: {str(e)}"

def setup_qdrant_collection():
    """Sets up a new Qdrant collection for this session."""
    global qdrant_client
    
    # Create a new in-memory client for each request to ensure clean state
    qdrant_client = QdrantClient(":memory:")
    
    # Create collection
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

# --- Request/Response Models ---

class QARequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --- Main Q&A Endpoint ---

@app.post("/qa", response_model=QAResponse)
async def question_answering(request: QARequest):
    """
    Main endpoint that takes a document URL and questions, processes them, and returns answers.
    """
    global embedding_model
    
    try:
        print(f"🔍 Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Initialize services for this request
        if embedding_model is None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model loaded")
        
        # Setup fresh Qdrant collection
        setup_qdrant_collection()
        print("✅ Qdrant collection initialized")
        
        # Step 1: Download and process the PDF
        print("📥 Downloading PDF...")
        pdf_bytes = download_pdf(request.documents)
        
        print("📄 Processing PDF...")
        doc_id = str(uuid.uuid4())
        processed_chunks = process_single_pdf_bytes(pdf_bytes, doc_id=doc_id)
        
        if not processed_chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from the PDF")
        
        print(f"✅ Extracted {len(processed_chunks)} chunks from PDF")
        
        # Step 2: Embed and index chunks
        print("🔍 Indexing chunks...")
        points = []
        for chunk in processed_chunks:
            vector = embedding_model.encode(chunk['content']).tolist()
            point_id = chunk['point_id']
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk['metadata'] | {"content": chunk['content']}
                )
            )
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"✅ Indexed {len(points)} chunks in Qdrant")
        
        # Step 3: Process each question
        answers = []
        for i, question in enumerate(request.questions):
            print(f"❓ Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            # Retrieve relevant chunks for this question
            relevant_chunks = retrieve_relevant_chunks(question, qdrant_client, embedding_model, top_k=8)
            
            # Generate answer using the retrieved context
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer.strip())
            
            print(f"✅ Generated answer {i+1}")
        
        print("🎉 All questions processed successfully!")
        
        return QAResponse(answers=answers)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ Error in Q&A processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Health Check Endpoint ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "embedding_model": embedding_model is not None,
            "gemini": GEMINI_API_KEY is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "🚀 RAG Q&A Service is running!",
        "endpoints": {
            "qa": "/qa - Main Q&A endpoint",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/qa",
            "body": {
                "documents": "URL to PDF document",
                "questions": ["List of questions to answer"]
            }
        }
    }

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global embedding_model
    try:
        print("🚀 Starting RAG Q&A Service...")
        print("✅ Service startup completed!")
    except Exception as e:
        print(f"❌ Startup error: {e}")

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)