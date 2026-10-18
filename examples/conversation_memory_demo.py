from ai_rag.agent.workflow import run
from ai_rag.memory.conversation_memory import ConversationMemory


def main():
    memory = ConversationMemory()

    run(
        "What is the status of CONN-1002?",
        memory=memory,
    )

    run(
        "What connection did I ask about?",
        memory=memory,
    )

    run(
        "What was its status?",
        memory=memory
    )

    run(
        "What is its status now?",
        memory=memory
    )

    run(
        "Remind me what happened to that connection.",
        memory=memory
    )

    run(
        "Has it recovered?",
        memory=memory
    )

    run(
        "Check it again.",
        memory=memory
    )

    run(
        "What was the status before?",
        memory=memory
    )

    run(
        "What's the status now?",
        memory=memory
    )


if __name__ == "__main__":
    main()