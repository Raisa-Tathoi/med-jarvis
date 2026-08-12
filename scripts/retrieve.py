from sentence_transformers import SentenceTransformer
from chroma_store import collection

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(question: str, n_results: int = 3) -> list[str]:
    question_embedding = model.encode(question)
    retrieved = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
    )
    return retrieved["documents"][0]


if __name__ == "__main__":
    question = input("Question: ")
    for doc in retrieve(question):
        print(doc)
        print("*****************")
