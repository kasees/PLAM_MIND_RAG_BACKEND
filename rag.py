import json
import os
import re
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

COLLECTION_NAME = "documents_collection"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
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
    query_vector = embed_model.encode(query).tolist()
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


def ask_rag_assistant(session_id: str, user_query: str) -> Dict[str, Any]:
    """Custom conversational RAG pipeline with LLM-based interview booking."""
    history = get_history(session_id)
    context = search_qdrant(user_query)

    system_instruction = f"""
You are a helpful AI assistant for Palm Mind Technology.
Answer user questions using ONLY the context below. If the answer is not in the context, state that clearly.

INTERVIEW BOOKING RULE:
If the user wants to book an interview, collect 4 required details: name, email, date, and time.
- If ANY of the 4 details are missing, politely ask for the missing details.
- Once ALL 4 details are provided, respond ONLY with valid JSON in this exact format (no markdown fences):
{{"booking": true, "name": "...", "email": "...", "date": "...", "time": "..."}}

Context from uploaded documents:
{context if context else "No documents uploaded yet."}
"""

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,  
        temperature=0.2,
    )


    reply = (completion.choices[0].message.content or "").strip()
    booking_saved = False

    if '"booking": true' in reply or '"booking":true' in reply:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply, flags=re.IGNORECASE).strip()
            data = json.loads(cleaned)
            save_interview_booking(
                session_id=session_id,
                name=data["name"],
                email=data["email"],
                date=data["date"],
                time=data["time"],
            )
            booking_saved = True
            reply = (
                f"Interview confirmed for {data['name']} on {data['date']} at {data['time']}. "
                f"Confirmation logged for {data['email']}."
            )
        except Exception:
            pass

    save_message(session_id, "user", user_query)
    save_message(session_id, "assistant", reply)

    return {
        "session_id": session_id,
        "reply": reply,
        "booking_confirmed": booking_saved,
    }
