from ai_rag.memory.extractor import extract_memory


def main():

    messages = [
        "Certificate CERT-1001 belongs to Partner A.",
        "I prefer React for frontend projects.",
        "I think Partner B may be moving to a new connection.",
        "Can you show me the certificates expiring this week?",
    ]

    for message in messages:

        print("\n" + "=" * 60)
        print("MESSAGE:")
        print(message)

        extraction = extract_memory(message)

        print("\nEXTRACTED MEMORY:")
        for fact in extraction.facts:
            print(fact.model_dump())


if __name__ == "__main__":
    main()