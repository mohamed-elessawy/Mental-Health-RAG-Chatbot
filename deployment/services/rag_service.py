from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from deployment.core.config import config

embedder = None
qdrant = None

def init_rag():
    global embedder, qdrant
    print("Initializing Sentence Transformer...")
    embedder = SentenceTransformer(config.EMBED_MODEL)
    
    print(f"Connecting to Qdrant at {config.QDRANT_URL}...")
    if config.QDRANT_API_KEY:
        qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    else:
        qdrant = QdrantClient(url=config.QDRANT_URL)
        
def retrieve_documents(query: str, collection_name: str = "mental_health_docs") -> list[str]:
    """Retrieves top_k similar documents from Qdrant."""
    if embedder is None or qdrant is None:
        raise RuntimeError("RAG Service not initialized. Call init_rag() during lifespan.")
        
    query_embedding = embedder.encode(query).tolist()
    

    search_result = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=config.RETRIEVAL_TOP_K
    )
    # Format the payload using the known schema ('question' and 'responses')
    documents = []
    for hit in search_result.points:
        question = hit.payload.get("question", "Unknown Question")
        responses = hit.payload.get("responses", "No responses available.")
        
        # Combine the user question and the expert response from the database
        doc_text = f"Similar Past Question: {question}\nSuggested Advice: {responses}"
        documents.append(doc_text)
        
    return documents
