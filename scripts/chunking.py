from extract_text import text

##the code here for chunking is taken from https://github.com/ALucek/chunking-strategies

import tiktoken

from chunking_evaluation.chunking import RecursiveTokenChunker
print("blurg")
##analyzing chunks
def analyze_chunks(chunks, use_tokens=False):
    print("blurg")
    # Print the chunks of interest
    print("\nNumber of Chunks:", len(chunks))
    print("\n", "="*50, "10th Chunk", "="*50,"\n", chunks[9])
    print("\n", "="*50, "11th Chunk", "="*50,"\n", chunks[10])
    
    chunk1, chunk2 = chunks[9], chunks[10]
    print("blurg")
    if use_tokens:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens1 = encoding.encode(chunk1)
        tokens2 = encoding.encode(chunk2)
        
        # Find overlapping tokens
        for i in range(len(tokens1), 0, -1):
            if tokens1[-i:] == tokens2[:i]:
                overlap = encoding.decode(tokens1[-i:])
                print("\n", "="*50, f"\nOverlapping text ({i} tokens):", overlap)
                return
        print("\nNo token overlap found")
    else:
        # Find overlapping characters
        for i in range(min(len(chunk1), len(chunk2)), 0, -1):
            if chunk1[-i:] == chunk2[:i]:
                print("\n", "="*50, f"\nOverlapping text ({i} chars):", chunk1[-i:])
                return
        print("\nNo character overlap found")
print("blurg")
recursive_character_chunker = RecursiveTokenChunker(
    chunk_size=800,  # Character Length
    chunk_overlap=0,  # Overlap
    length_function=len,  # Character length with len()
    separators=["\n\n", "\n", ".", "?", "!", " ", ""] # According to Research
)
print("blurg")
recursive_character_overlap_chunks = recursive_character_chunker.split_text(text)
analyze_chunks(recursive_character_overlap_chunks, use_tokens=False)

print("blurg")