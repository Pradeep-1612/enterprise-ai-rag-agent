from ai_rag.memory.conversation_memory import ConversationMemory

messages = ConversationMemory()

messages.add(
    "user",
    "Check certificate-1001."
)

messages.add(
    "assistant",
    "Certificate-1001 has 10 days remaining."
)

messages.add(
    "user",
    "What connection is it using?"
)

print(messages.get_recent())