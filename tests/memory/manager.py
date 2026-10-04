from ai_rag.memory.manager import MemoryManager


memory = MemoryManager()

memory.remember(
    "certificate-1001",
    "belongs_to",
    "Partner A"
)

memory.remember(
    "certificate-1001",
    "connected_to",
    "CONN-1002"
)

print("All memory:")
print(memory.all_facts())

print("\nCertificate relationships:")
print(
    memory.retrieve(
        subject="certificate-1001"
    )
)

print("\nConnections:")
print(
    memory.retrieve(
        predicate="connected_to"
    )
)

memory.update(
    "certificate-1001",
    "connected_to",
    "CONN-1003"
)

print(memory.all_facts())

memory.forget(
    "certificate-1001",
    "connected_to"
)

print(memory.all_facts())