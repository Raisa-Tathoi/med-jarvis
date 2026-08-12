import chromadb

## https://www.geeksforgeeks.org/nlp/introduction-to-chromadb/

chroma_client = chromadb.PersistentClient(
    path="database/chroma_db"
)
collection = chroma_client.get_or_create_collection(name="first_aid_usmle")


if __name__ == "__main__":
    from embed import embeddings
    from chunking import recursive_character_overlap_chunks

    collection.add(
        documents=recursive_character_overlap_chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": "First Aid for USMLE Step 2",
             "chapter": "All"}
        ] * len(recursive_character_overlap_chunks),
        ids=[f"chunk_{i}" for i in range(len(recursive_character_overlap_chunks))]
    )
    print(f"Ingested {len(recursive_character_overlap_chunks)} chunks into 'first_aid_usmle'.")
