class ConversationMemory:
    def __init__(self):
        self.messages = []
        self.facts = []

    def add(self, role, content):
        self.messages.append({
            "role": role,
            "content": content
        })

    def add_facts(self, tool_results, decision):
        self.facts.append({
            "tool_results": tool_results,
            "decision": decision
        })

    def get_recent(self, limit=10):
        return self.messages[-limit:]

    def get_recent_facts(self, limit=10):
        return self.facts[-limit:]

    def clear(self):
        self.messages = []
        self.facts = []