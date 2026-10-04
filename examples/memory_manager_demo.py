from ai_rag.memory.manager import MemoryManager
from ai_rag.memory.models import MemoryFact


def main():

    memory = MemoryManager()

    fact = MemoryFact(
    type="fact",
    subject="CERT-1001",
    predicate="belongs_to",
    object="Partner A",
    source="user",
    confidence="verified",
)

    print("\n=== REMEMBER ===")

    memory.remember(fact)

    print(memory.all_facts())

    print("\n=== EXACT RETRIEVE ===")

    results = memory.retrieve(
        subject="CERT-1001"
    )

    print(results)

    print("\n=== SEMANTIC SEARCH ===")

    results = memory.search(
        "Which partner owns certificate CERT-1001?"
    )

    print(results)

    print("\n=== UPDATE ===")

    updated_fact = MemoryFact(
    type="fact",
    subject="CERT-1001",
    predicate="belongs_to",
    object="Partner B",
    source="user",
    confidence="verified",
)

    memory.update(updated_fact)

    print(memory.all_facts())

    print("\n=== SEMANTIC SEARCH AFTER UPDATE ===")

    results = memory.search(
        "Which partner owns certificate CERT-1001?"
    )

    print(results)

    print("\n=== FORGET ===")

    memory.forget(
        subject="CERT-1001",
        predicate="belongs_to",
    )

    print(memory.all_facts())

    print("\n=== SEMANTIC SEARCH AFTER FORGET ===")

    results = memory.search(
        "Which partner owns certificate CERT-1001?"
    )

    print(results)


if __name__ == "__main__":
    main()