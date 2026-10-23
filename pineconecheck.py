import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder

load_dotenv(find_dotenv())

# 1. Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "hybrid-search-index")

pc = Pinecone(api_key=api_key)

# 2. Check and create index if needed (must be dotproduct for hybrid search)
existing_indexes = [idx.name for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating hybrid index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully!")

index = pc.Index(index_name)

# 3. Dense Embeddings (sentence-transformers)
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Sparse Encoder (BM25 for keyword/lexical search)
bm25_encoder = BM25Encoder().default()

# 5. Connect Pinecone Hybrid Search Retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embed_model,
    sparse_encoder=bm25_encoder,
    index=index,
    top_k=4,
    alpha=0.5  # 0.5 balances dense (semantic) and sparse (keyword) search
)

# 6. Example Texts to add
sentences = [
    "Atomic habits are small changes that compound into remarkable results.",
    "The 4 laws of behavior change are make it obvious, attractive, easy, and satisfying.",
    "The habit loop consists of cue, craving, response, and reward.",
]

if __name__ == "__main__":
    print("Adding texts to Pinecone Hybrid index...")
    retriever.add_texts(sentences)
    print("Texts added!")

    # 7. Querying the hybrid retriever
    query = "What are the four laws of behavior change?"
    print(f"\nSearching for: '{query}'")
    results = retriever.invoke(query)

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content)
