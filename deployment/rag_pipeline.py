import os
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

groq_client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant          = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = "mental_health_docs"


def rewrite_query(user_message: str) -> tuple:
    prompt = f"""You are helping a mental health support chatbot find relevant information.

Given the user message, do two things:

1. Extract personal context: note gender, relationship, or specific situation.
   If none found, write "none".

2. Write a focused search query that:
   - Removes names and irrelevant personal details
   - Keeps the core emotional and mental health concern
   - Expands vague terms into descriptive emotional language
   - Is between 15 and 30 words

Respond in this exact format:
CONTEXT: <personal context>
QUERY: <search query>

User message: "{user_message}"
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    output       = response.choices[0].message.content.strip()
    context_line = ""
    query_line   = ""
    for line in output.split("\n"):
        if line.startswith("CONTEXT:"):
            context_line = line.replace("CONTEXT:", "").strip()
        if line.startswith("QUERY:"):
            query_line = line.replace("QUERY:", "").strip()
    return query_line, context_line


def retrieve(search_query: str, top_k: int = 3) -> list:
    query_vector = embedding_model.encode(search_query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points
    return [
        {
            "score"    : hit.score,
            "question" : hit.payload["question"],
            "responses": hit.payload["responses"],
            "topics"   : hit.payload["topics"]
        }
        for hit in results
    ]


def generate_response(user_message: str, retrieved: list,
                      emotion: str = "neutral",
                      personal_context: str = "none") -> str:
    context = "\n\n".join(
        f"Counselor response {i+1}.{j+1}:\n{r}"
        for i, doc in enumerate(retrieved)
        for j, r in enumerate(doc["responses"])
    )
    prompt = f"""You are a compassionate mental health support assistant.

User emotional state : {emotion}
Personal context     : {personal_context}

Guidelines:
- Use the counseling examples below as reference for tone and approach.
- Always be empathetic, non-judgmental, and supportive.
- Use personal context to personalize your response
- Do NOT diagnose. Do NOT prescribe medication.
- If user seems in crisis, recommend professional help

Reference counseling examples:
{context}

User message: {user_message}

Your response:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def rag_answer(user_message: str, emotion: str = "neutral") -> dict:
    search_query, personal_context = rewrite_query(user_message)
    retrieved    = retrieve(search_query, top_k=3)
    answer       = generate_response(user_message, retrieved, emotion, personal_context)
    return {
        "user_message"    : user_message,
        "emotion"         : emotion,
        "search_query"    : search_query,
        "personal_context": personal_context,
        "answer"          : answer,
        "sources"         : [r["question"][:80] for r in retrieved]
    }