from sentence_transformers import SentenceTransformer
from chunking import recursive_character_overlap_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(recursive_character_overlap_chunks)

print(len(embeddings)) ## should be the number of chunks I have