import ollama

from ai_rag.memory.models import MemoryExtraction, MemoryFact


def extract_memory(message: str) -> MemoryExtraction:

    prompt = f"""
You are a memory extraction component for an enterprise
certificate-management assistant.

Read the user message and extract useful explicit information
that may be relevant to future conversations.

Extract only information explicitly stated by the user.

Do not:
- invent information
- infer unstated information
- add conversational filler
- determine whether information is verified

If the user expresses uncertainty, classify the memory as
"claim".

Allowed memory types:
- fact
- claim
- preference
- context

For each memory, provide:
- type
- subject
- predicate
- object

Examples:

User:
"Certificate CERT-1001 belongs to Partner A."

Memory:
subject = "CERT-1001"
predicate = "belongs_to"
object = "Partner A"
type = "fact"

User:
"I prefer React for frontend projects."

Memory:
subject = "user"
predicate = "prefers"
object = "React"
type = "preference"

If there is no useful memory, return an empty facts list.

Return ONLY valid JSON.

User message:
{message}
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format=MemoryExtraction.model_json_schema(),
    )

    content = response["message"]["content"]
    print("\nRAW LLM OUTPUT:")
    print(content)

    extraction = MemoryExtraction.model_validate_json(
        content
    )

    return MemoryExtraction(
        facts=[
            MemoryFact(
                type=fact.type,
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                source="user",
                confidence="unverified",
            )
            for fact in extraction.facts
        ]
    )