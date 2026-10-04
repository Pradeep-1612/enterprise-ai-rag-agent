import os
import json

import chromadb
import ollama

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from ai_rag.tools.definitions import tools_schema
from ai_rag.tools.models import tool_registry


load_dotenv()


# ---------------------------------------------------------
# CLIENTS / MODELS
# ---------------------------------------------------------

# OpenAI client is currently unused because we are using
# local embeddings + Ollama.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

chroma = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma.get_collection(
    name="enterprise_docs"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# QUESTION
# ---------------------------------------------------------

question = (
    "The certificate has 10 days remaining and is connected "
    "to CONN-1002. What should we do?"
)  # Success case

# question = "Evaluate the certificate with remaining days set to abc."  # Failure case


# ---------------------------------------------------------
# OPENAI EMBEDDING - CURRENTLY DISABLED
# ---------------------------------------------------------

# 1. Convert question into an embedding
# response = client.embeddings.create(
#     model="text-embedding-3-small",
#     input=question
# )

# question_embedding = response.data[0].embedding


# ---------------------------------------------------------
# AGENT STATE
# ---------------------------------------------------------

state = {
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


# ---------------------------------------------------------
# NODE: RETRIEVE
# ---------------------------------------------------------

def retrieve(state):
    print("\n[NODE] RETRIEVE")

    question_embedding = embedding_model.encode(
        state["question"]
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    contexts = results["documents"][0]

    state["context"] = "\n\n---\n\n".join(contexts)

    state["steps"].append("retrieved_policy")

    state["next_node"] = "prepare_agent"

    return state


# ---------------------------------------------------------
# NODE: PREPARE_AGENT
# ---------------------------------------------------------

def prepare_agent(state):
    print("\n[NODE] PREPARE_AGENT")

    state["messages"] = [
        {
            "role": "user",
            "content": f"""
You are an enterprise certificate management assistant.

You have two tools:

1. evaluate_certificate
   - MUST be called whenever the question provides the number
     of days remaining.

2. get_connection_status
   - MUST be called whenever the question provides a connection ID.

IMPORTANT TOOL EXECUTION RULES:

- If the question contains both remaining days and a connection ID,
  call BOTH tools.
- Call all applicable tools BEFORE generating a final answer.
- Tool calls MUST be made using the provided tool-calling interface.
- Do NOT write tool calls as JSON or plain text.
- Do NOT generate a final answer until all applicable tools have returned.
- Tool results are authoritative.
- Do not contradict tool results.
- Do not infer that a conditional action is currently required.
- Only recommend actions explicitly supported by the policy or tool results.

Retrieved policy context:
{state["context"]}

Question:
{state["question"]}
"""
        }
    ]

    state["steps"].append("prepared_agent")

    state["next_node"] = "agent"

    return state


# ---------------------------------------------------------
# NODE: AGENT
# ---------------------------------------------------------

def agent(state):
    print("\n[NODE] AGENT")

    # Safety boundary: maximum LLM reasoning iterations.
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
        # Agent says it has enough information.
        state["next_node"] = "route_after_tools"

    state["steps"].append("agent")

    return state


# ---------------------------------------------------------
# NODE: TOOLS
# ---------------------------------------------------------

def tools(state):
    print("\n[NODE] TOOLS")

    last_message = state["messages"][-1]

    tool_calls = last_message.get(
        "tool_calls",
        []
    )

    for call in tool_calls:

        # Safety boundary: maximum tool executions.
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

    # Continue the agent loop.
    state["next_node"] = "agent"

    return state


# ---------------------------------------------------------
# NODE: ROUTE_AFTER_TOOLS
#
# Deterministic business-rule evaluation.
#
# This node does NOT generate the final answer.
# It produces a structured decision.
# ---------------------------------------------------------

def route_after_tools(state):
    print("\n[NODE] ROUTE_AFTER_TOOLS")

    certificate_result = None
    connection_result = None

    for item in state["tool_results"]:

        if item["tool"] == "evaluate_certificate":
            certificate_result = item["result"]

        elif item["tool"] == "get_connection_status":
            connection_result = item["result"]

    # -----------------------------------------------------
    # Certificate evaluation failed
    # -----------------------------------------------------

    if (
        not certificate_result
        or not certificate_result["success"]
    ):

        state["decision"] = {
            "action": "unable_to_evaluate",
            "reason": "Certificate evaluation failed."
        }

        state["steps"].append(
            "route_after_tools"
        )

        state["next_node"] = "final_agent"

        return state

    certificate = certificate_result["data"]

    # -----------------------------------------------------
    # 1. Emergency takes highest priority
    # -----------------------------------------------------

    if certificate["emergency_renewal"]:

        state["decision"] = {
            "action": "emergency_renewal",
            "reason": (
                "Certificate has fewer than 7 days remaining."
            )
        }

    # -----------------------------------------------------
    # 2. Certificate requires notification
    # -----------------------------------------------------

    elif certificate["notify_operations"]:

        if (
            connection_result
            and connection_result["success"]
        ):

            connection = connection_result["data"]

            if connection["status"] == "failed":

                state["decision"] = {
                    "action": "human_approval",
                    "reason": (
                        "Certificate requires operations "
                        "notification and the associated "
                        "connection is failed."
                    )
                }

            else:

                state["decision"] = {
                    "action": "notify_operations",
                    "reason": (
                        "Certificate has fewer than "
                        "15 days remaining."
                    )
                }

        else:

            state["decision"] = {
                "action": "notify_operations",
                "reason": (
                    "Certificate has fewer than "
                    "15 days remaining."
                )
            }

    # -----------------------------------------------------
    # 3. Nothing special required
    # -----------------------------------------------------

    else:

        state["decision"] = {
            "action": "no_action",
            "reason": (
                "Certificate does not currently require "
                "emergency renewal or operations notification."
            )
        }

    state["steps"].append(
        "route_after_tools"
    )

    state["next_node"] = "final_agent"

    return state


# =========================================================
# LEGACY IMPLEMENTATION
# =========================================================
#
# This was the previous architecture where
# ROUTE_AFTER_TOOLS directly routed to action nodes.
#
# Kept here for learning/reference.
# =========================================================

# def route_after_tools(state):
#     print("\n[NODE] ROUTE_AFTER_TOOLS")
#
#     certificate_result = None
#     connection_result = None
#
#     for item in state["tool_results"]:
#         if item["tool"] == "evaluate_certificate":
#             certificate_result = item["result"]
#
#         elif item["tool"] == "get_connection_status":
#             connection_result = item["result"]
#
#     # Certificate evaluation failed
#     if not certificate_result or not certificate_result["success"]:
#         state["final_answer"] = "Unable to evaluate the certificate."
#         state["next_node"] = "final"
#         return state
#
#     certificate = certificate_result["data"]
#
#     # 1. Emergency takes highest priority
#     if certificate["emergency_renewal"]:
#         state["next_node"] = "emergency"
#         return state
#
#     # 2. Certificate requires notification
#     if certificate["notify_operations"]:
#
#         # Check connection status
#         if connection_result and connection_result["success"]:
#             connection = connection_result["data"]
#
#             if connection["status"] == "failed":
#                 state["next_node"] = "human_approval"
#                 return state
#
#         state["next_node"] = "notify"
#         return state
#
#     # 3. Nothing special required
#     state["final_answer"] = (
#         "The certificate does not currently require "
#         "emergency renewal or operations notification."
#     )
#
#     state["next_node"] = "final"
#
#     return state


# =========================================================
# LEGACY ACTION NODES
# =========================================================
#
# These are no longer part of the active workflow.
# They are kept for comparison with the previous design.
# =========================================================

# def notify(state):
#     print("\n[NODE] NOTIFY")
#
#     state["steps"].append("notify_operations")
#
#     # Mocking the notifications being sent
#     state["notification"] = {
#         "recipient": "operations",
#         "reason": (
#             "Certificate has fewer than 15 days remaining."
#         )
#     }
#
#     state["final_answer"] = (
#         "The certificate has fewer than 15 days remaining, "
#         "so the operations team should be notified."
#     )
#
#     state["next_node"] = "final"
#
#     return state


# def emergency(state):
#     print("\n[NODE] EMERGENCY")
#
#     state["steps"].append("emergency_renewal")
#
#     state["next_node"] = "final"
#
#     state["final_answer"] = (
#         "Emergency certificate renewal is required."
#     )
#
#     return state


# def human_approval(state):
#     print("\n[NODE] HUMAN_APPROVAL")
#
#     state["steps"].append("human_approval")
#
#     state["approval_required"] = True
#
#     state["final_answer"] = (
#         "The certificate has fewer than 15 days remaining, "
#         "and the associated connection is currently failed. "
#         "Human approval is required before proceeding."
#     )
#
#     state["next_node"] = "final"
#
#     return state


# ---------------------------------------------------------
# NODE: FINAL_AGENT
#
# Converts the deterministic decision + evidence into
# the final natural-language response.
# ---------------------------------------------------------

def final_agent(state):
    print("\n[NODE] FINAL_AGENT")

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
- Do not change the action.
- Do not add actions that are not present in the decision.
- Do not reinterpret tool results.
- Do not claim validation occurred unless a tool explicitly returned it.
- Explain the decision using the actual tool results.
- Keep the response concise.
- If the decision is "human_approval", state that human approval is required.
- Do NOT ask the user to approve, review, authorize, or take the action.
- Do NOT imply that approval has been granted.
"""
    })

    response = ollama.chat(
        model="llama3.2:3b",
        messages=state["messages"]
    )

    state["steps"].append(
        "final_agent"
    )

    state["final_answer"] = (
        response["message"]["content"]
    )

    state["next_node"] = "final"

    return state


# ---------------------------------------------------------
# NODE: FINAL
# ---------------------------------------------------------

def final(state):
    print("\n[NODE] FINAL")

    print("\nAnswer:")
    print(state["final_answer"])

    print("\nSteps:")
    print(state["steps"])

    return state


# ---------------------------------------------------------
# NODE REGISTRY
# ---------------------------------------------------------
#
# Only ACTIVE nodes belong here.
#
# notify, emergency, and human_approval are intentionally
# not registered because they belong to the legacy design.
# ---------------------------------------------------------

nodes = {
    "retrieve": retrieve,
    "prepare_agent": prepare_agent,
    "agent": agent,
    "tools": tools,
    "route_after_tools": route_after_tools,
    "final_agent": final_agent,
    "final": final
}


# ---------------------------------------------------------
# WORKFLOW EXECUTION
# ---------------------------------------------------------

while state["next_node"] != "final":

    current_node = state["next_node"]

    state = nodes[current_node](state)


state = final(state)