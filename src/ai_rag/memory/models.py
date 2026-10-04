from typing import Literal
from pydantic import BaseModel


class MemoryFact(BaseModel):
    type: Literal[
        "fact",
        "claim",
        "preference",
        "context"
    ]

    subject: str
    predicate: str
    object: str

    source: Literal[
        "user",
        "tool",
        "database",
        "document",
        "agent"
    ]

    confidence: Literal[
        "verified",
        "unverified",
        "inferred"
    ]


class MemoryExtraction(BaseModel):
    facts: list[MemoryFact]