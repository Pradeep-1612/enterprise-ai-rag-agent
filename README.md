# Enterprise AI RAG Agent

**Enterprise AI Engineering · Agentic AI · LLM Engineering · RAG · AI Agents & Tool Calling · Conversation & Semantic Memory · Python · Workflow Orchestration · Deterministic Decisioning**

An AI-powered enterprise operations assistant exploring how to investigate certificate lifecycle risks, check enterprise connection health, and support operational decisions through natural-language interactions.

Built around a core engineering principle:

> **Use the LLM for reasoning, and keep state, business rules, and execution under application control.**

## What does it do?

Enterprise operations often require engineers to retrieve documentation, inspect operational data, interpret results, and determine the appropriate next action. This project explores how an AI agent can coordinate these activities while preserving application-level control over business decisions.

The agent supports the following workflows:

- **Certificate lifecycle assessment:** Evaluate certificate validity and identify certificates requiring attention.
- **Connection health checks:** Retrieve the operational status of enterprise connections.
- **Rule-based decisioning:** Apply deterministic business rules to operational results and determine the appropriate next action.
- **Contextual conversations:** Reference earlier questions and results through conversation history and structured memory.
- **Knowledge-grounded responses:** Retrieve relevant documentation using RAG to support responses with enterprise context.
- **Tool-driven execution:** Translate natural-language requests into validated tool calls rather than relying solely on generated answers.

### Example interactions

| User request | Agent behavior |
|---|---|
| “Evaluate certificate CONN-1002.” | Invokes certificate evaluation and applies the configured decision rules. |
| “What is the status of CONN-1002?” | Retrieves the connection's operational status. |
| “What was its status before?” | Uses available conversation history or structured records to answer historically. |
| “What is its status now?” | Should retrieve fresh operational data rather than assume the previous result remains current. |
| “Which action should we take?” | Uses available tool results and deterministic decisioning to produce the applicable outcome. |

These examples describe the intended interaction patterns; actual results depend on the tools, data, and business rules configured in the local implementation.

## Architecture

```text
                       User Request
                            │
                            ▼
                   Intent Classification
                            │
                            ▼
                    Workflow Orchestration
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
             RAG           Memory        Agent
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                     Tool Orchestration
                            │
                            ▼
                    Pydantic Validation
                            │
                            ▼
                     Tool Execution
                            │
                            ▼
                  Deterministic Decisioning
                            │
                            ▼
                     Final Response
```

The current implementation uses a custom Python workflow to coordinate retrieval, memory, agent execution, tool calls, routing, and response generation.

The workflow maintains explicit application state and bounds execution through configured iteration and tool-call limits.

## Engineering decisions

### 1. Separate reasoning from business decisions

The LLM interprets requests and assists with tool selection. Application code evaluates operational data against defined business rules.

This prevents generated reasoning from independently authorizing business actions or inventing operational outcomes.

### 2. Validate tool inputs

Pydantic models validate tool arguments before execution, establishing a structured boundary between model-generated requests and application operations.

### 3. Distinguish historical memory from current state

Conversation history and structured facts help answer follow-up questions. Current operational questions require fresh tool results rather than treating historical data as authoritative.

### 4. Ground responses in retrieved information

Sentence Transformers generate document embeddings, and ChromaDB provides vector storage and retrieval for the RAG pipeline.

### 5. Keep components independently testable

Retrieval, memory, tool execution, routing, and decisioning are separate concerns, making their behavior easier to inspect and evaluate.

## Technology stack

| Area | Technologies |
|---|---|
| Language | Python |
| LLM runtime | Ollama |
| Language models | Llama 3.x |
| Embeddings | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Input validation | Pydantic |
| Agent workflow | Custom Python orchestration |
| Evaluation | Agent and route-decision evaluation examples |

The project is designed for local experimentation without requiring a paid LLM API.

## Run locally

### Prerequisites

- Python version supported by `pyproject.toml`
- [Ollama](https://ollama.com/download)
- Git

### 1. Clone and install

```bash
git clone <repository-url>
cd Enterprise-AI-RAG-Agent

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Replace `<repository-url>` with the actual repository URL.

### 2. Start Ollama

On macOS, open the Ollama application. On Linux, start the Ollama service or run:

```bash
ollama serve
```

If Ollama is already running, there is no need to start another instance.

### 3. Download the configured model

For example, with Llama 3.2 3B:

```bash
ollama pull llama3.2:3b
```

If the code is configured for Llama 3.1 8B, use:

```bash
ollama pull llama3.1:8b
```

The model name in the source code must match an installed model. Verify available models with:

```bash
ollama list
```

### 4. Run the main agent

From the repository root, with the virtual environment activated:

```bash
python -m ai_rag.agent.workflow
```

This launches the main workflow and allows you to explore the configured agent interactions.

### 5. Run the examples

The `examples/` directory contains six standalone demonstrations.

| Example | Focus |
|---|---|
| `memory_demo.py` | Foundational memory behavior |
| `conversation_memory_demo.py` | Conversational context and follow-up questions |
| `memory_manager_demo.py` | Memory management |
| `memory_extractor_demo.py` | Memory and fact extraction |
| `agent_evaluation_demo.py` | Agent evaluation |
| `route_decision_evaluation_demo.py` | Routing and decision evaluation |

Run individual examples from the repository root:

```bash
python -m examples.memory_demo
python -m examples.conversation_memory_demo
python -m examples.memory_manager_demo
python -m examples.memory_extractor_demo
python -m examples.agent_evaluation_demo
python -m examples.route_decision_evaluation_demo
```

Examples that invoke Ollama require the runtime and configured model. Examples that use vector retrieval may also require an existing ChromaDB collection and indexed documents.

The exact behavior and prerequisites are defined by each example's implementation.

## Project structure

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

## Engineering principles

- **Controlled autonomy:** LLM reasoning operates within application-defined execution boundaries.
- **Deterministic outcomes:** Business decisions are made by explicit rules, not generated text alone.
- **Validated interfaces:** Model-generated tool arguments are validated before execution.
- **Context-aware memory:** Historical information is distinguished from fresh operational data.
- **Local-first development:** Models can run locally through Ollama without a paid inference API.

## Author

**Pradeep Reddy Juturu**

Lead Software Engineer | AI Engineering | Full-Stack | Software Architecture

- **Portfolio:** https://pradeepreddyj-portfolio.vercel.app/
- **GitHub:** https://github.com/Pradeep-1612
- **LinkedIn:** https://www.linkedin.com/in/pradeep-reddy-juturu/