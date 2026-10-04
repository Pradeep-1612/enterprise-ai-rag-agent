from ai_rag.memory.extractor import extract_memory
from ai_rag.memory.semantic import SemanticMemory


memory = SemanticMemory()


message = """
Certificate-1001 belongs to Partner A and is connected
to CONN-1002.
"""

extracted = extract_memory(message)

for fact in extracted.facts:
    memory.index(
        fact.model_dump()
    )


query = "Which connection does Partner A use?"

results = memory.search(query)


print("\nRetrieved memories:")

for document, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print("\nMemory:", document)
    print("Metadata:", metadata)
    print("Distance:", distance)