from ai_rag.memory.manager import MemoryManager
from ai_rag.memory.extractor import extract_memory


memory = MemoryManager()

message = """
Certificate-1001 belongs to Partner A and is connected
to CONN-1002.
"""

extracted = extract_memory(message)

print("Extracted:")
print(extracted)

for fact in extracted.facts:
    memory.remember(fact)

print("\nStored memory:")
print(memory.all_facts())