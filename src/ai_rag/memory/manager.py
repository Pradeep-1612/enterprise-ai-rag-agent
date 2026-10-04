from copy import deepcopy

from ai_rag.memory.models import MemoryFact
from ai_rag.memory.semantic import SemanticMemory


class MemoryManager:

    def __init__(self):
        self.memory = {
            "facts": []
        }
        self.semantic_memory = SemanticMemory()

    def remember(self, fact: MemoryFact):
        """
        Store a validated memory fact.

        Exact duplicate facts are not stored twice.
        """

        fact_data = fact.model_dump()

        if fact_data not in self.memory["facts"]:
            self.memory["facts"].append(fact_data)

            self.semantic_memory.index(fact_data)

    def retrieve(
        self,
        subject=None,
        predicate=None,
        object=None,
        memory_type=None
    ):
        """
        Retrieve memories matching the supplied criteria.
        Any omitted criterion acts as a wildcard.
        """

        results = []

        for fact in self.memory["facts"]:

            if subject is not None:
                if fact["subject"] != subject:
                    continue

            if predicate is not None:
                if fact["predicate"] != predicate:
                    continue

            if object is not None:
                if fact["object"] != object:
                    continue

            if memory_type is not None:
                if fact["type"] != memory_type:
                    continue

            results.append(deepcopy(fact))

        return results
    
    def search(self, query, n_results=5):
        """
        Search memories using semantic similarity.
        """

        return self.semantic_memory.search(
            query,
            n_results
        )

    def update(self, fact: MemoryFact):
        """
        Replace an existing memory with the same
        subject + predicate.
        """

        existing_facts = [
            existing
            for existing in self.memory["facts"]
            if (
                existing["subject"] == fact.subject
                and existing["predicate"] == fact.predicate
            )
        ]

        for existing in existing_facts:
            self.semantic_memory.delete(existing)

        self.memory["facts"] = [
            existing
            for existing in self.memory["facts"]
            if not (
                existing["subject"] == fact.subject
                and existing["predicate"] == fact.predicate
            )
        ]

        self.remember(fact)

    def forget(self, subject, predicate=None):
        """
        Forget either all memories for a subject
        or a specific subject/predicate relationship.
        """

        if predicate is None:

            facts_to_remove = [
                fact
                for fact in self.memory["facts"]
                if fact["subject"] == subject
            ]

        else:

            facts_to_remove = [
                fact
                for fact in self.memory["facts"]
                if (
                    fact["subject"] == subject
                    and fact["predicate"] == predicate
                )
            ]

        for fact in facts_to_remove:
            self.semantic_memory.delete(fact)

        self.memory["facts"] = [
            fact
            for fact in self.memory["facts"]
            if fact not in facts_to_remove
        ]

    def all_facts(self):
        return deepcopy(self.memory["facts"])