from ai_rag.agent.state import create_initial_state

from ai_rag.agent.nodes import (
    retrieve,
    prepare_agent,
    agent,
    tools,
    route_after_tools,
    final_agent,
    final
)


nodes = {
    "retrieve": retrieve,
    "prepare_agent": prepare_agent,
    "agent": agent,
    "tools": tools,
    "route_after_tools": route_after_tools,
    "final_agent": final_agent,
    "final": final
}


def run(question):

    state = create_initial_state(
        question
    )

    while state["next_node"] != "final":

        current_node = state["next_node"]

        state = nodes[current_node](
            state
        )

    return final(state)


if __name__ == "__main__":

    question = (
        "The certificate has 10 days remaining "
        "and is connected to CONN-1002. "
        "What should we do?"
    )

    run(question)