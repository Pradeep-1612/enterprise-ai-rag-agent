import json

import ollama

from pydantic import ValidationError

from ai_rag.tools.definitions import tools_schema
from ai_rag.tools.models import tool_registry

from ai_rag.rag.retriever import Retriever


retriever = Retriever()


def retrieve(state):
    print("\n[NODE] RETRIEVE")

    state["context"] = retriever.retrieve_context(
        state["question"]
    )

    state["steps"].append(
        "retrieved_policy"
    )

    state["next_node"] = "prepare_agent"

    return state

def prepare_agent(state):
    print("\n[NODE] PREPARE_AGENT")

    memory = state["conversation_memory"]
    history = memory.get_recent(limit=10)

    state["messages"] = [
        {
            "role": "system",
            "content": f"""
You are an enterprise certificate management assistant.

Follow these tool execution rules:
- Call evaluate_certificate when remaining days are provided.
- Call get_connection_status when a connection ID is provided.
- Call all applicable tools before generating a final answer.
- Tool results are authoritative; do not contradict them.
- Only recommend actions supported by policy or tool results.

CONVERSATION RULES:
- Use conversation history to resolve references in follow-up questions.
- If history contains enough information to answer a question,
  answer directly without calling tools unnecessarily.
- Call tools when the question requires current system data
  or a new business evaluation.
- Do not repeat a previous business decision unless relevant
  to the current question.

Retrieved policy context:
{state["context"]}
"""
        },
        *history,
        {
            "role": "user",
            "content": state["question"]
        }
    ]

    state["steps"].append("prepared_agent")

    state["next_node"] = (
        "agent"
        if state.get("intent") == "operational"
        else "final_agent"
    )

    return state


def agent(state):
    print("\n[NODE] AGENT")

    if state["iteration"] >= state["max_iterations"]:

        state["decision"] = {
            "action": "agent_execution_stopped",
            "reason": (
                "The agent reached the maximum number of reasoning "
                "iterations and stopped safely."
            )
        }

        state["next_node"] = "final_agent"

        return state

    state["iteration"] += 1

    print(
        f"Agent iteration: {state['iteration']}"
    )

    response = ollama.chat(
        model="llama3.2:3b",
        messages=state["messages"],
        tools=tools_schema
    )

    state["messages"].append(
        response["message"]
    )

    tool_calls = response["message"].get(
        "tool_calls"
    )

    if tool_calls:
        state["next_node"] = "tools"
    else:
        state["next_node"] = "route_after_tools"

    state["steps"].append(
        "agent"
    )

    return state


def tools(state):
    print("\n[NODE] TOOLS")

    last_message = state["messages"][-1]

    tool_calls = last_message.get(
        "tool_calls",
        []
    )

    for call in tool_calls:

        if state["tool_calls"] >= state["max_tool_calls"]:

            state["decision"] = {
                "action": "tool_execution_stopped",
                "reason": (
                    "The agent reached the maximum allowed "
                    "tool-call limit and stopped safely."
                )
            }

            state["next_node"] = "final_agent"

            return state

        state["tool_calls"] += 1

        print(
            f"Tool iteration: {state['tool_calls']}"
        )

        tool_name = call["function"]["name"]
        arguments = call["function"]["arguments"]

        tool_config = tool_registry.get(
            tool_name
        )

        # -------------------------------------------------
        # Unknown tool
        # -------------------------------------------------

        if not tool_config:

            result = {
                "success": False,
                "error": "UNKNOWN_TOOL",
                "message": (
                    f"Tool '{tool_name}' is not registered."
                )
            }

            state["tool_results"].append({
                "tool": tool_name,
                "result": result
            })

            state["steps"].append(
                "(Tool Error) " + tool_name
            )

            state["messages"].append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result)
            })

            continue

        # -------------------------------------------------
        # Validate tool arguments
        # -------------------------------------------------

        try:

            args = tool_config["args_model"](
                **arguments
            )

        except ValidationError as e:

            result = {
                "success": False,
                "error": "INVALID_ARGUMENTS",
                "message": (
                    "Tool arguments failed validation."
                ),
                "details": e.errors()
            }

            state["tool_results"].append({
                "tool": tool_name,
                "result": result
            })

            state["steps"].append(
                "(Tool Error) " + tool_name
            )

            state["messages"].append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result)
            })

            continue

        # -------------------------------------------------
        # Execute tool
        # -------------------------------------------------

        try:

            result = tool_config["function"](
                **args.model_dump()
            )

            result = {
                "success": True,
                "data": result
            }

        except Exception as e:

            result = {
                "success": False,
                "error": "TOOL_EXECUTION_FAILED",
                "message": str(e)
            }

        # -------------------------------------------------
        # Store tool result
        # -------------------------------------------------

        state["tool_results"].append({
            "tool": tool_name,
            "result": result
        })

        state["steps"].append(
            "(Tool) " + tool_name
        )

        state["messages"].append({
            "role": "tool",
            "tool_name": tool_name,
            "content": json.dumps(result)
        })

    state["next_node"] = "agent"

    return state

def final_agent(state):
    print("\n[NODE] FINAL_AGENT")

    if state.get("intent") == "conversation":
        prompt = f"""
    Answer the user's question using conversation history
    and the structured operational records.

    Conversation history:
    {state["conversation_memory"].get_recent(limit=10)}

    Structured operational records:
    {state["conversation_memory"].get_recent_facts(limit=10)}

    Rules:
    - Prefer structured tool results over generated assistant prose
    when answering questions about observed system state.
    - For historical questions, report the recorded historical fact.
    - Do not claim historical data represents the current state.
    - Do not call tools or invent information.

    User question:
    {state["question"]}
    """
        messages = [
            {
                "role": "system",
                "content": "You are a concise enterprise assistant."
            },
            {"role": "user", "content": prompt}
        ]
    else:
        state["messages"].append({
            "role": "user",
            "content": f"""
                The information-gathering phase is complete.

                Original question:
                {state["question"]}

                Tool results:
                {json.dumps(state["tool_results"], indent=2)}

                Deterministic business decision:
                {json.dumps(state["decision"], indent=2)}

                Generate the final response to the user.

                STRICT RULES:
                - Treat the deterministic business decision as authoritative.
                - Do not change the action or add unsupported actions.
                - Do not reinterpret tool results.
                - Do not claim validation occurred unless a tool explicitly returned it.
                - Explain the decision using actual tool results.
                - Keep the response concise.
                - If the decision is "human_approval", state that human approval is required.
                - Do NOT ask the user to approve, review, authorize, or take the action.
                - Do NOT imply that approval has been granted.
                """
        })
        messages = state["messages"]

    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages
    )

    state["steps"].append("final_agent")
    state["final_answer"] = response["message"]["content"]
    state["next_node"] = "final"

    return state


def final(state):
    print("\n[NODE] FINAL")

    print("\nAnswer:")
    print(state["final_answer"])

    print("\nSteps:")
    print(state["steps"])

    return state