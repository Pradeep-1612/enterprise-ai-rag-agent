def create_initial_state(question):
    return {
        "question": question,
        "context": "",
        "messages": [],
        "tool_results": [],
        "decision": None,
        "steps": [],
        "next_node": "retrieve",
        "final_answer": "",

        # Agent execution budgets
        "iteration": 0,
        "max_iterations": 5,

        "tool_calls": 0,
        "max_tool_calls": 10
    }