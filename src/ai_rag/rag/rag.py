import os
import chromadb
import ollama

from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ai_rag.tools.definitions import tools_schema
from ai_rag.tools.models import tool_registry
from pydantic import ValidationError
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

chroma = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma.get_collection(
    name="enterprise_docs"
)

question = "The certificate has 10 days remaining and is connected to CONN-1002. What should we do?" # Success case
# question = "Evaluate the certificate with remaining days set to abc." # Failure case

# # 1. Convert question into an embedding
# response = client.embeddings.create(
#     model="text-embedding-3-small",
#     input=question
# )

# question_embedding = response.data[0].embedding

state = {
    "question": question,
    "context": "",
    "messages": [],
    "tool_results": [],
    "steps": [],
    "next_node": "retrieve"
}



embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


question_embedding = embedding_model.encode(question).tolist()

# 2. Search vector database
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"],
    # where={"topic": "risk"}
)

# 3. Extract retrieved documents
contexts = results["documents"][0]

context = "\n\n---\n\n".join(contexts)

print("Retrieved context:")
print(context)


state["context"] = context;
state["steps"].append("retrieved_policy")


for i in range(len(results["documents"][0])):
    print("\n--- Result", i + 1, "---")
    print("Distance:", results["distances"][0][i])
    print("Source:", results["metadatas"][0][i]["source"])
    print("Topic:", results["metadatas"][0][i]["topic"])
    print("Text:", results["documents"][0][i])



# -------------------------
# 4. LLM GENERATION
# -------------------------

# prompt = f"""
# You are an enterprise certificate management assistant.

# Answer the user's question using ONLY the provided context.

# If the answer cannot be found in the context,
# say: "I don't have enough information."

# Context:
# {context}

# Question:
# {question}
# """

# -------------------------
# 5. Create messages
# -------------------------

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

response = None

while True:

    response = ollama.chat(
        model="llama3.2:3b",
        messages=state["messages"],
        tools=tools_schema
    )

    print(response)

    state["messages"].append(response["message"])

    tool_calls = response["message"].get("tool_calls")

    if not tool_calls:
        state["steps"].append("generated_answer")
        break

    for call in tool_calls:

        tool_name = call["function"]["name"]
        arguments = call["function"]["arguments"]

        tool_config = tool_registry.get(tool_name)

        if not tool_config:
            result = {
                "success": False,
                "error": "UNKNOWN_TOOL",
                "message": f"Tool '{tool_name}' is not registered."
            }

            state["messages"].append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result)
            })

            continue

        try:
            args = tool_config["args_model"](**arguments)

        except ValidationError as e:
            result = {
                "success": False,
                "error": "INVALID_ARGUMENTS",
                "message": "Tool arguments failed validation.",
                "details": e.errors()
            }

            state["messages"].append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result)
            })

            continue

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

        state["tool_results"].append({"tool": tool_name, "result": result})
        state["steps"].append(tool_name)
        state["messages"].append({
            "role": "tool",
            "tool_name": tool_name,
            "content": json.dumps(result)
        })

print("\nAnswer:")
print(response["message"]["content"])

print("\nAgent state:")
print(json.dumps(state["tool_results"], indent=2))
print(state["steps"])