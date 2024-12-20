# requirements.txt
"""
fastapi
uvicorn
chromadb
langchain
python-multipart
pydantic
torch
transformers
numpy
"""

# main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import chromadb
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import chardet  # Add this import for encoding detection

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # Using cosine similarity
)

def detect_encoding(content: bytes) -> str:
    """Detect the encoding of the content using chardet"""
    result = chardet.detect(content)
    return result['encoding'] or 'utf-8'

def safe_decode(content: bytes) -> str:
    """Safely decode content with fallback encodings"""
    # First try detected encoding
    detected_encoding = detect_encoding(content)
    try:
        return content.decode(detected_encoding)
    except UnicodeDecodeError:
        # Fallback encodings to try
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with errors='ignore'
        return content.decode('utf-8', errors='ignore')

def chunk_by_sentences(input_text: str, tokenizer: AutoTokenizer) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Split text into sentences using tokenizer"""
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    
    return chunks, span_annotations

def late_chunking(model_output: torch.Tensor, span_annotation: list, max_length=None):
    """Generate embeddings using late chunking technique"""
    token_embeddings = model_output[0]
    outputs = []
    
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if max_length is not None:
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)
    
    return outputs

class Query(BaseModel):
    question: str

# @app.post("/upload")
# async def upload_document(file: UploadFile = File(...)):
#     content = await file.read()
#     text = content.decode("utf-8")
    
#     # Generate chunks using sentence-based splitting
#     chunks, span_annotations = chunk_by_sentences(text, tokenizer)
    
#     # Generate embeddings using late chunking
#     inputs = tokenizer(text, return_tensors='pt')
#     model_output = model(**inputs)
#     embeddings = late_chunking(model_output, [span_annotations])[0]
    
#     # Convert embeddings to list format for ChromaDB
#     embeddings_list = [emb.tolist() for emb in embeddings]
    
#     # Add to ChromaDB
#     collection.add(
#         embeddings=embeddings_list,
#         documents=chunks,
#         ids=[f"doc_{i}" for i in range(len(chunks))]
#     )
    
#     return {"message": f"Processed {len(chunks)} chunks with late embeddings"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # Try to detect and decode the file content
        text = safe_decode(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or corrupted")
        
        # Generate chunks using sentence-based splitting
        chunks, span_annotations = chunk_by_sentences(text, tokenizer)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text chunks could be extracted")
        
        # Generate embeddings using late chunking
        inputs = tokenizer(text, return_tensors='pt')
        model_output = model(**inputs)
        embeddings = late_chunking(model_output, [span_annotations])[0]
        
        # Convert embeddings to list format for ChromaDB
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Add to ChromaDB
        collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            ids=[f"doc_{i}" for i in range(len(chunks))]
        )
        
        return {
            "message": f"Successfully processed {len(chunks)} chunks",
            "encoding_used": detect_encoding(content),
            "chunk_lengths": [len(chunk) for chunk in chunks]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}. Please ensure the file is a valid text document."
        )


@app.post("/query")
async def query_documents(query: Query):
    # Generate query embedding
    query_inputs = tokenizer(query.question, return_tensors='pt')
    query_output = model(**query_inputs)
    query_embedding = query_output[0].mean(dim=1).detach().cpu().numpy()[0]
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=2
    )
    
    return {
        "context": results['documents'][0],
        "metadata": results['metadatas'][0] if results['metadatas'] else None
    }

# Optional: Add endpoint to clear the collection
@app.post("/clear")
async def clear_collection():
    collection.delete(where={})
    return {"message": "Collection cleared"}