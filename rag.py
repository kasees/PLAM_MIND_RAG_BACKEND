import json
import os
import uuid
from typing import Any, Dict, List
import redis
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from database import save_interview_booking

load_dotenv()

COLLECTION_NAME="documents_collection"
embed_model=SentenceTransfomer("all-MiniLM-L6-v2")
qdrant = QdrantClient(path="./qdrant_data")
redis_db = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def store_chunks_in_qdrant(chunks: List[str], filename: str) -> None:
    """Generates embeddings and stores text chunks in Qdrant."""
    vectors = embed_model.encode(chunks).tolist()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": chunk, "filename": filename},
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def search_qdrant(query: str, top_k: int = 3) -> str:
    """Retrieves top-k relevant document chunks from Qdrant."""
    query_vector = embed_model.encode([query]).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    matched = [hit.payload["text"] for hit in results if hit.payload]
    return "\n\n".join(matched)


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Fetches multi-turn chat history from Redis."""
    messages = redis_db.lrange(f"chat:{session_id}", 0, -1)
    return [json.loads(m) for m in messages]


def save_message(session_id: str, role: str, content: str) -> None:
    """Appends a chat turn to Redis memory."""
    redis_db.rpush(f"chat:{session_id}", json.dumps({"role": role, "content": content}))
    redis_db.ltrim(f"chat:{session_id}", -10, -1)
