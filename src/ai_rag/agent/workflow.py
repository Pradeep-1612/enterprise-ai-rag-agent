from ai_rag.agent.state import create_initial_state

from ai_rag.agent.nodes import (
    retrieve,
    prepare_agent,
    agent,
    tools,
    final_agent,
    final
)

from ai_rag.agent.decisioning import route_after_tools

from ai_rag.memory.conversation_memory import ConversationMemory

from ai_rag.agent.intent import classify_intent


nodes = {
    "retrieve": retrieve,
    "prepare_agent": prepare_agent,
    "agent": agent,
    "tools": tools,
    "route_after_tools": route_after_tools,
    "final_agent": final_agent,
    "final": final
}


def run(question, memory=None):
    if memory is None:
        memory = ConversationMemory()

    history = {
        "messages": memory.get_recent(limit=10),
        "facts": memory.get_recent_facts(limit=10)
    }
    intent = classify_intent(question, history)

    state = create_initial_state(question)
    state["conversation_memory"] = memory
    state["intent"] = intent
    print(f"[DEBUG] intent={state.get('intent')!r}")
    state["next_node"] = (
        "retrieve" if intent == "operational"
        else "prepare_agent"
    )

    # Workflow loop
    while state["next_node"] != "final":
        current_node = state["next_node"]
        state = nodes[current_node](state)

    result = final(state)

    memory.add("user", question)
    memory.add("assistant", state["final_answer"])

    tool_results = state.get("tool_results", [])
    decision = state.get("decision")

    if tool_results or decision:
        memory.add_facts(
            tool_results=tool_results,
            decision=decision,
        )

    return result


if __name__ == "__main__":

    question = (
        "The certificate has 10 days remaining "
        "and is connected to CONN-1002. "
        "What should we do?"
    )

    run(question)