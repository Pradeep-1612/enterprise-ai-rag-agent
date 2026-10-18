
from typing import Literal

import ollama
from pydantic import BaseModel


class Intent(BaseModel):
    name: Literal["conversation", "operational", "clarification"]


def classify_intent(question, history):
    response = ollama.chat(
        model="llama3.1:8b",
        format=Intent.model_json_schema(),
        messages=[
            {
                "role": "system",
                "content": """
Classify the user's request:
- conversation: answer from dialogue history and stored facts, including historical status questions.
- operational: requires tools, current data, or business evaluation
- clarification: essential information is missing or ambiguous

Use history to resolve references.
A request for current status is operational.
""",
            },
            {
                "role": "user",
                "content": f"History: {history}\nQuestion: {question}",
            },
        ],
    )

    return Intent.model_validate_json(
        response["message"]["content"]
    ).name
