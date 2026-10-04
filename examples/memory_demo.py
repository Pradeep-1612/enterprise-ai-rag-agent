from ai_rag.memory.manager import MemoryManager
from ai_rag.memory.models import MemoryFact


def main():

    manager = MemoryManager()

    fact = MemoryFact(
        type="fact",
        subject="CERT-1001",
        predicate="belongs_to",
        object="Partner A",
        source="user",
        confidence="verified",
    )

    print("\n--- Remembering ---")

    manager.remember(fact)

    print("\n--- Exact retrieval ---")

    exact_results = manager.retrieve(
        subject="CERT-1001",
        predicate="belongs_to"
    )

    print(exact_results)

    print("\n--- Semantic search ---")

    semantic_results = manager.search(
        "Which partner owns certificate CERT-1001?"
    )

    print(semantic_results)


if __name__ == "__main__":
    main()