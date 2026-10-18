# Enterprise AI RAG Agent

A hands-on exploration of enterprise AI agent engineering, combining Retrieval-Augmented Generation (RAG), conversation memory, tool orchestration, structured validation, and deterministic business decisioning.

The goal is to understand how to build AI systems where the LLM provides reasoning while the application retains control over state, execution, and business rules.

> **Core principle:** Use the LLM for reasoning, and keep state, business rules, and execution under application control.

## 1. Architecture

```text
                       User Request
                            │
                            ▼
                  ┌───────────────────┐
                  │  Agent Workflow   │
                  │    workflow.py    │
                  └─────────┬─────────┘
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
             Intent       Memory         RAG
           Classification  │             │
               │           │             │
               └────────────┼────────────┘
                            ▼
                     Agent Reasoning
                            │
                            ▼
                    Tool Orchestration
                            │
                            ▼
                  Deterministic Decisioning
                            │
                            ▼
                     Final Response
```

The application controls the workflow and determines which operations execute. The LLM assists with interpreting requests, generating responses, and selecting tools where appropriate.

## 2. Technology stack

| Component | Technology |
|---|---|
| Language | Python |
| Local LLM runtime | Ollama |
| Language models | Llama 3.x |
| Embedding model | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Schema validation | Pydantic |
| Workflow orchestration | Python-based custom workflow |
| Testing and evaluation | Project-specific evaluation examples |

## 3. Prerequisites

Install:

- Python compatible with the version declared in `pyproject.toml`
- [Ollama](https://ollama.com/download)
- Git

The project runs locally without requiring a paid LLM API.

## 4. Installation

### Clone the repository

```bash
git clone <repository-url>
cd Enterprise-AI-RAG-Agent
```

Replace `<repository-url>` with the actual repository URL.

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Run commands from the repository root unless an example specifies otherwise.

## 5. Configure Ollama

### Start Ollama

On macOS, open the Ollama application. On Linux, start the Ollama service or run:

```bash
ollama serve
```

If Ollama is already running, do not start another server.

Verify the installation:

```bash
ollama --version
ollama list
```

### Download the configured model

For the lightweight Llama 3.2 model:

```bash
ollama pull llama3.2:3b
```

If the code is configured to use Llama 3.1 8B instead:

```bash
ollama pull llama3.1:8b
```

The model name in the code must match an installed model. Check the configuration in `workflow.py`, `intent.py`, and other modules that invoke Ollama.

Test inference independently:

```bash
ollama run llama3.2:3b
```

Enter a prompt, verify the response, and type `/bye` to exit.

## 6. Run the main agent

With the virtual environment activated and Ollama running:

```bash
python -m ai_rag.agent.workflow
```

The workflow is the primary entry point. It coordinates retrieval, memory, agent reasoning, tool execution, decision routing, and response generation.

## 7. Explore the examples

The `examples/` directory contains six Python demonstrations. Run each from the repository root.

### Example catalogue

| Example | Learning objective |
|---|---|
| `memory_demo.py` | Understand the foundational memory implementation. |
| `conversation_memory_demo.py` | Explore conversational context and follow-up questions. |
| `memory_manager_demo.py` | Explore memory management and retrieval. |
| `memory_extractor_demo.py` | Examine extracting useful information from conversations. |
| `agent_evaluation_demo.py` | Explore evaluation of agent behavior. |
| `route_decision_evaluation_demo.py` | Examine workflow routing and decision evaluation. |

These descriptions are based on the example names; consult each file for its actual implementation and dependencies.

### Run individual examples

```bash
python -m examples.memory_demo
python -m examples.conversation_memory_demo
python -m examples.memory_manager_demo
python -m examples.memory_extractor_demo
python -m examples.agent_evaluation_demo
python -m examples.route_decision_evaluation_demo
```

If an example is not importable as a Python module, try direct execution:

```bash
python examples/memory_demo.py
```

Run one example at a time to isolate errors and understand its behavior.

### Recommended learning sequence

1. `memory_demo.py` — start with the simplest memory concepts.
2. `conversation_memory_demo.py` — understand conversational follow-ups.
3. `memory_manager_demo.py` — explore memory organization and retrieval.
4. `memory_extractor_demo.py` — examine fact extraction.
5. `route_decision_evaluation_demo.py` — understand routing and decision evaluation.
6. `agent_evaluation_demo.py` — explore broader agent evaluation.
7. `python -m ai_rag.agent.workflow` — connect the capabilities in the complete workflow.

The sequence is a suggested learning path, not a statement about execution dependencies.

## 8. RAG and vector storage

The project uses Sentence Transformers to create embeddings and ChromaDB to store and retrieve document representations.

```text
Enterprise Documents
        │
        ▼
Embedding Model
        │
        ▼
ChromaDB Collection
        │
        ▼
Similarity Retrieval
        │
        ▼
Retrieved Context
        │
        ▼
Agent Response
```

Relevant directories:

- `documents/` — source documents.
- `chroma_db/` — root vector database artifacts.
- `examples/chroma_db/` — example-specific vector database artifacts, if used by the example code.
- `src/ai_rag/rag/` — retrieval implementation.

Check the ingestion process and configured database path before running RAG-dependent examples. Do not assume the index is automatically created when the agent starts.

## 9. Conversation memory

The project explores how an agent can use earlier interactions without confusing historical information with current operational state.

### Conversation history

Recent messages provide context for follow-up questions and references to previously discussed entities.

### Structured memory

Structured facts can preserve tool results and deterministic business decisions. This makes prior outcomes easier to reference without relying entirely on an LLM's recollection of a conversation.

### Semantic memory

Semantic retrieval can find relevant information by similarity rather than requiring an exact phrase match. Its behavior depends on the implemented storage and retrieval pipeline.

### Memory design principle

Historical facts and current operational state are different:

- Historical questions should use the relevant stored record.
- Requests for current status should retrieve fresh operational data.
- A previous tool result should not be treated as proof of the current state.

## 10. Tool orchestration

Current tools include:

- Certificate lifecycle evaluation.
- Enterprise connection status lookup.

Tool arguments are validated with Pydantic before execution.

```text
Agent
  │
  ▼
Tool Selection
  │
  ▼
Argument Validation
  │
  ▼
Tool Execution
  │
  ▼
Tool Result
  │
  ▼
Application Decision
```

The application should distinguish invalid arguments, tool execution failures, and valid business results. A successful tool invocation does not necessarily mean the business operation succeeded.

## 11. Deterministic decisioning

Business rules remain in application code.

The LLM can help identify the information required for a decision, but the application evaluates the returned data against defined rules.

For example, certificate lifecycle decisions can use remaining validity, connection status, and configured thresholds to determine the next action.

```text
Validated Operational Data
           │
           ▼
    Business Rules
           │
           ▼
  Deterministic Outcome
           │
           ▼
     Final Response
```

The response should reflect the actual decision produced by the application. The LLM must not invent approvals or operational outcomes that the decision logic did not return.

## 12. Project structure

```text
Enterprise-AI-RAG-Agent/
├── src/
│   └── ai_rag/
│       ├── agent/
│       │   ├── workflow.py
│       │   ├── nodes.py
│       │   ├── state.py
│       │   ├── intent.py
│       │   └── decisioning.py
│       ├── rag/
│       ├── memory/
│       └── tools/
├── examples/
│   ├── agent_evaluation_demo.py
│   ├── conversation_memory_demo.py
│   ├── memory_demo.py
│   ├── memory_extractor_demo.py
│   ├── memory_manager_demo.py
│   ├── route_decision_evaluation_demo.py
│   └── chroma_db/
├── documents/
├── chroma_db/
├── tests/
├── pyproject.toml
└── README.md
```

## 13. Troubleshooting

| Problem | Checks |
|---|---|
| `ollama: command not found` | Install Ollama and verify the executable is on your `PATH`. |
| Connection refused | Start the Ollama application or service. |
| Model not found | Run `ollama list` and pull the configured model. |
| `ModuleNotFoundError: ai_rag` | Activate `.venv` and install the project with `pip install -e .`. |
| Example import errors | Run from the repository root and check the example's module structure. |
| No documents retrieved | Verify document ingestion, collection name, and ChromaDB path. |
| Unexpected memory responses | Inspect stored messages, structured facts, and whether the request needs historical or fresh operational data. |

## 14. Roadmap

- [x] Conversation memory foundations
- [x] RAG integration foundations
- [x] Tool orchestration and argument validation
- [x] Deterministic business decisioning
- [ ] LangGraph-based workflow orchestration
- [ ] Agent evaluation and regression tests
- [ ] Guardrails and policy enforcement
- [ ] Model Context Protocol (MCP)
- [ ] Human-in-the-loop approvals
- [ ] Observability and tracing
- [ ] Persistent memory and workflow checkpointing

## 15. Engineering principle

An LLM should be a reasoning component inside a well-engineered system—not the system itself.

Retrieval, memory, validation, execution, workflow state, and business decisions should remain explicit and independently testable.

## Author

**Pradeep Reddy Juturu**

Lead Software Engineer | AI Engineering | Full-Stack | Software Architecture

- Portfolio: https://pradeepreddyj-portfolio.vercel.app/
- GitHub: https://github.com/Pradeep-1612
- LinkedIn: https://www.linkedin.com/in/pradeep-reddy-juturu/
