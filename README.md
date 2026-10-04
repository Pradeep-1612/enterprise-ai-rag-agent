# Enterprise-AI-RAG-Agent

A production-oriented exploration of **enterprise AI agent architecture** combining RAG, memory, tool orchestration, and deterministic decisioning.

The core objective is simple:

> **Use the LLM for reasoning, and keep state, business rules, and execution under application control.**

## Architecture

```text
                         User Request
                              │
                              ▼
                     ┌─────────────────┐
                     │  Agent Workflow │
                     │   workflow.py   │
                     └────────┬────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
             RAG            Memory            Tools
              │               │                │
              └───────────────┼────────────────┘
                              ▼
                     Decision / Routing
                              │
                              ▼
                       Final Response
```

## Agent Workflow

`workflow.py` is the main agent entry point.

It orchestrates the execution flow:

```text
RETRIEVE
   ↓
PREPARE_AGENT
   ↓
AGENT
   ↓
TOOLS ─────────┐
   ↓           │
ROUTE          │
   └───────────┘
   ↓
FINAL_AGENT
   ↓
FINAL
```

The workflow maintains explicit state, controls tool execution, bounds iterations, and separates **LLM reasoning from deterministic application decisions**.

Run the agent with:

```bash
python3 -m ai_rag.agent.workflow
```

## Core Capabilities

### RAG

Grounds agent responses using document retrieval with:

* Sentence Transformers
* `all-MiniLM-L6-v2`
* ChromaDB

### Memory

Combines:

* **Structured memory** for authoritative state
* **Semantic memory** for similarity-based retrieval
* LLM-assisted memory extraction

### Tools

Current tools include:

* Certificate lifecycle evaluation
* Enterprise connection status

Tool arguments are validated with Pydantic before execution.

### Deterministic Decisioning

Business rules remain in application code.

The LLM can reason about **what needs to happen**; deterministic code decides **what is actually allowed to happen**.

## Technology

**Python · Ollama · Llama 3.x · Sentence Transformers · ChromaDB · Pydantic**

The project runs locally without a paid LLM API.

## Project Structure

```text
Enterprise-AI-RAG-Agent/
├── src/ai_rag/
│   ├── agent/
│   │   ├── workflow.py       # Agent orchestration
│   │   ├── nodes.py
│   │   └── state.py
│   │
│   ├── rag/
│   ├── memory/
│   └── tools/
│
├── examples/
├── documents/
├── chroma_db/
├── tests/
├── pyproject.toml
└── README.md
```

## Getting Started

```bash
git clone <repository-url>
cd Enterprise-AI-RAG-Agent

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -e .

ollama pull llama3.2:3b
```

Run the agent:

```bash
python3 -m ai_rag.agent.workflow
```

## Engineering Focus

This project explores a fundamental principle of enterprise AI:

**An LLM should be a reasoning component inside a well-engineered system — not the system itself.**

The architecture therefore keeps **retrieval, memory, validation, tools, workflow state, and business decisions** explicit and independently controllable.

## Roadmap

* Conversation memory
* RAG + memory integration
* Agent evaluation
* Guardrails
* MCP
* Human-in-the-loop
* Observability
* Production persistence

## Author

**Pradeep Reddy Juturu**
Lead Software Engineer | AI Engineering | Full-Stack | Software Architecture

[Portfolio](https://pradeepreddyj-portfolio.vercel.app/) · [GitHub](https://github.com/Pradeep-1612) · [LinkedIn](https://www.linkedin.com/in/pradeep-reddy-juturu/)
