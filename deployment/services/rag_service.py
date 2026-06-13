import logging

import litellm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from deployment.core.config import config
from deployment.schemas.prompts import (
    get_generation_system_prompt,
    get_query_rewrite_prompt,
)

logger = logging.getLogger("serenity.services.rag")

embedder = None
qdrant = None


def init_rag():
    global embedder, qdrant
    logger.info("Initializing Sentence Transformer...")
    embedder = SentenceTransformer(config.EMBED_MODEL)
    logger.info("Connecting to Qdrant...")
    qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def rewrite_query(user_message: str) -> tuple:
    response = litellm.completion(
        model=config.INTENT_LLM_MODEL,
        messages=[{"role": "user", "content": get_query_rewrite_prompt(user_message)}],
        temperature=0,
    )
    output = response.choices[0].message.content.strip()
    context_line = ""
    query_line = ""
    for line in output.split("\n"):
        if line.startswith("CONTEXT:"):
            context_line = line.replace("CONTEXT:", "").strip()
        if line.startswith("QUERY:"):
            query_line = line.replace("QUERY:", "").strip()
    return query_line, context_line


def retrieve_documents(query: str) -> list:
    if embedder is None or qdrant is None:
        raise RuntimeError("RAG not initialized. Call init_rag() first.")
    query_embedding = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name="mental_health_docs",
        query=query_embedding,
        limit=config.RETRIEVAL_TOP_K,
    ).points
    return [
        {
            "question": hit.payload.get("question", ""),
            "responses": hit.payload.get("responses", []),
            "topics": hit.payload.get("topics", []),
        }
        for hit in results
    ]


def generate_response(
    user_message: str,
    retrieved: list,
    emotion: str = "neutral",
    personal_context: str = "none",
    history: list | None = None,
) -> str:

    context = "\n\n".join(
        f"Counselor response {i + 1}.{j + 1}:\n{r}"
        for i, doc in enumerate(retrieved)
        for j, r in enumerate(doc["responses"][:3])
    )

    system_prompt = get_generation_system_prompt(emotion, personal_context, context)

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = litellm.completion(
        model=config.GENERATION_LLM_MODEL, messages=messages, temperature=0.7
    )
    return response.choices[0].message.content.strip()


def rag_answer(
    user_message: str, emotion: str = "neutral", history: list | None = None
) -> dict:
    search_query, personal_context = rewrite_query(user_message)
    retrieved = retrieve_documents(search_query)
    answer = generate_response(
        user_message=user_message,
        retrieved=retrieved,
        emotion=emotion,
        personal_context=personal_context,
        history=history,
    )
    return {
        "answer": answer,
        "search_query": search_query,
        "personal_context": personal_context,
        "sources": [r["question"][:80] for r in retrieved],
    }
