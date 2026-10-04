# Enterprise AI Agent — RAG, Memory & Tool Orchestration

A production-oriented AI engineering project exploring how to build an enterprise AI agent that combines **Retrieval-Augmented Generation (RAG), structured memory, semantic memory, tool use, and deterministic business rules**.

The project is intentionally built from the ground up to understand the engineering patterns behind modern AI agents rather than hiding the architecture behind a framework.

> **Status:** Work in progress — core RAG, tool orchestration, agent workflow, and memory foundations are implemented.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Agent Workflow    │
                         └──────────┬───────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
        ┌───────────────┐   ┌────────────────┐   ┌───────────────┐
        │     RAG       │   │     Memory     │   │    Tools      │
        │               │   │                │   │               │
        │ Vector Search │   │ Structured     │   │ Certificate   │
        │ + Embeddings  │   │ + Semantic     │   │ Evaluation    │
        │               │   │ + Conversation │   │ Connection    │
        └───────┬───────┘   └───────┬────────┘   │ Status        │
                │                   │            └───────┬───────┘
                └───────────────────┼────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Deterministic        │
                         │ Decision Layer       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Final Agent      │
                         │ Grounded Response    │
                         └──────────────────────┘
```

---

## What This Project Explores

### 1. Retrieval-Augmented Generation

Documents are:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
```

When a user asks a question:

```text
Question
   ↓
Question Embedding
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

The project uses:

* **Sentence Transformers** for local embeddings
* `all-MiniLM-L6-v2` embedding model
* **ChromaDB** as the local vector database
* **Ollama** for local LLM inference

The goal is to understand that retrieval and generation are separate concerns. A retrieved document does not automatically make an answer correct.

---

## 2. Tool-Based Reasoning

The agent can invoke deterministic application tools instead of asking the LLM to perform business logic.

Current tools include:

### Certificate Evaluation

Evaluates certificate lifecycle conditions such as:

* Remaining validity
* Emergency renewal requirement
* Operations notification requirement
* Post-renewal validation requirements

Example:

```python
evaluate_certificate(remaining_days=10)
```

returns structured information such as:

```json
{
  "remaining_days": 10,
  "at_risk": true,
  "notify_operations": true,
  "emergency_renewal": false
}
```

### Connection Status

Retrieves the current status of an enterprise connection.

```python
get_connection_status("CONN-1002")
```

Example:

```json
{
  "status": "failed",
  "trading_partner": "Partner B"
}
```

---

## 3. Deterministic Business Decisions

One of the key architectural principles in this project is:

> **LLMs interpret and reason over information; deterministic application code owns business rules.**

For example, the LLM should not decide whether:

```text
remaining_days < 7
```

means emergency renewal.

That rule belongs to application code.

The agent can use the LLM to determine which tools are relevant, but the resulting business decision is evaluated by deterministic code.

This creates a separation between:

```text
LLM
Semantic understanding
Tool selection
Natural-language reasoning
        │
        ▼
Application
Business rules
Validation
Authorization
Deterministic decisions
```

---

# Memory Architecture

The project implements two complementary forms of memory.

```text
                    MemoryManager
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
      Structured Memory       Semantic Memory
              │                     │
        Exact retrieval       Similarity search
              │                     │
          Python data           ChromaDB
```

## Structured Memory

Structured memories are represented using a validated Pydantic model:

```python
class MemoryFact(BaseModel):
    type: Literal[
        "fact",
        "claim",
        "preference",
        "context"
    ]

    subject: str
    predicate: str
    object: str

    source: Literal[
        "user",
        "tool",
        "database",
        "document",
        "agent"
    ]

    confidence: Literal[
        "verified",
        "unverified",
        "inferred"
    ]
```

Example:

```json
{
  "type": "fact",
  "subject": "CERT-1001",
  "predicate": "belongs_to",
  "object": "Partner A",
  "source": "user",
  "confidence": "unverified"
}
```

Structured memory supports:

* Exact retrieval
* Updates
* Deletion
* Duplicate prevention
* Structured filtering

---

## Semantic Memory

Structured memories are also indexed semantically.

For example:

```text
CERT-1001 belongs_to Partner A
```

is converted into an embedding and stored in ChromaDB.

This allows queries such as:

```text
"Which partner owns certificate CERT-1001?"
```

to retrieve the relevant memory even when the wording does not exactly match the stored fact.

The semantic index is treated as a **secondary index**, not the source of truth.

---

## Memory Lifecycle

The `MemoryManager` owns the memory lifecycle:

```text
remember()
    │
    ▼
Structured Memory
    │
    └──────► Semantic Index

retrieve()
    │
    └──────► Exact structured lookup

search()
    │
    └──────► Semantic similarity search

update()
    │
    ├──────► Remove old semantic index
    └──────► Store new memory

forget()
    │
    ├──────► Remove structured memory
    └──────► Remove semantic index
```

This keeps callers independent from the underlying storage implementation.

The application interacts with:

```python
MemoryManager
```

rather than directly interacting with ChromaDB.

---

# LLM-Based Memory Extraction

The project also explores using an LLM to identify useful memories from conversational input.

Example:

```text
Certificate CERT-1001 belongs to Partner A.
```

can be transformed into:

```json
{
  "type": "fact",
  "subject": "CERT-1001",
  "predicate": "belongs_to",
  "object": "Partner A"
}
```

The extractor deliberately does **not** determine:

```text
source
confidence
```

Those are application-level attributes.

For example:

```text
LLM
 │
 ├── type
 ├── subject
 ├── predicate
 └── object
        │
        ▼
Application
 │
 ├── source = user
 └── confidence = unverified
```

This prevents the LLM from arbitrarily deciding whether a piece of information should be trusted.

---

# Agent Workflow

The agent workflow currently follows an explicit state-machine pattern.

```text
START
  │
  ▼
RETRIEVE
  │
  ▼
PREPARE_AGENT
  │
  ▼
AGENT
  │
  ├────── Tool Calls ──────┐
  │                        │
  │                        ▼
  │                     TOOLS
  │                        │
  │                        ▼
  │                ROUTE_AFTER_TOOLS
  │                        │
  │                        ▼
  └──────────────────── AGENT
                           │
                           ▼
                     FINAL_AGENT
                           │
                           ▼
                         FINAL
```

The workflow maintains explicit state including:

* User question
* Retrieved context
* Messages
* Tool results
* Decisions
* Execution steps
* Iteration count
* Tool-call limits
* Final answer

The workflow currently uses a lightweight custom orchestration layer rather than immediately introducing an agent framework.

This makes the underlying orchestration patterns explicit before potentially moving to a framework such as LangGraph.

---

# Tool Registry

Tools are registered with their corresponding argument schemas.

Example:

```python
tool_registry = {
    "evaluate_certificate": {
        "function": evaluate_certificate,
        "args_model": EvaluateCertificateArgs,
    },
    "get_connection_status": {
        "function": get_connection_status,
        "args_model": ConnectionStatusArgs,
    },
}
```

Tool arguments are validated with Pydantic before execution.

This provides a boundary between:

```text
LLM-generated tool arguments
             ↓
       Pydantic validation
             ↓
      Application function
```

The LLM therefore cannot directly bypass the application's expected argument contract.

---

# Technology Stack

| Component          | Technology                 |
| ------------------ | -------------------------- |
| Language           | Python                     |
| LLM Runtime        | Ollama                     |
| LLM                | Llama 3.x                  |
| Embeddings         | Sentence Transformers      |
| Embedding Model    | `all-MiniLM-L6-v2`         |
| Vector Database    | ChromaDB                   |
| Data Validation    | Pydantic                   |
| Package Management | setuptools / pip           |
| Environment        | Python virtual environment |

The project is designed to run locally without requiring a paid LLM API.

---

# Project Structure

```text
ai-rag/
│
├── src/
│   └── ai_rag/
│       │
│       ├── agent/
│       │   ├── state.py
│       │   ├── nodes.py
│       │   └── workflow.py
│       │
│       ├── memory/
│       │   ├── models.py
│       │   ├── manager.py
│       │   ├── extractor.py
│       │   ├── semantic.py
│       │   └── conversation_memory.py
│       │
│       ├── rag/
│       │   ├── ingest.py
│       │   ├── retriever.py
│       │   ├── rag.py
│       │   └── rag_graph_based.py
│       │
│       └── tools/
│           ├── functions.py
│           ├── definitions.py
│           └── models.py
│
├── examples/
│   ├── memory_manager_demo.py
│   └── memory_extractor_demo.py
│
├── documents/
├── chroma_db/
├── tests/
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Running Locally

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai-rag
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python3 -m pip install -e .
```

## 4. Install Ollama

Install Ollama for your operating system and make sure it is running.

Then pull a local model:

```bash
ollama pull llama3.2:3b
```

A stronger local model can be used depending on available system resources.

## 5. Run the examples

Memory manager:

```bash
python3 -m examples.memory_manager_demo
```

Memory extraction:

```bash
python3 -m examples.memory_extractor_demo
```

---

# Design Principles

### 1. LLMs are not the source of truth

LLMs are probabilistic.

Critical business decisions should therefore be validated and enforced by deterministic application code.

---

### 2. Retrieval and generation are different problems

A good RAG system needs to reason about:

* Chunking
* Embedding quality
* Similarity search
* Top-K selection
* Metadata filtering
* Context size
* Reranking
* Generation quality

Retrieving a document is not equivalent to producing a correct answer.

---

### 3. Memory should have explicit semantics

Not every conversational statement should become a trusted fact.

The project distinguishes:

```text
fact
claim
preference
context
```

and:

```text
verified
unverified
inferred
```

This provides a foundation for more reliable enterprise memory.

---

### 4. Keep the source of truth separate from indexes

Semantic memory is an index over structured memory.

The vector database should not become the authoritative representation of business state merely because it is convenient for retrieval.

---

### 5. Validate model-generated data

LLM-generated tool arguments and memory objects should cross a validation boundary before entering application logic.

Pydantic is currently used for this boundary.

---

### 6. Avoid unnecessary abstraction

The project intentionally avoids building a complex ontology or relationship engine for simple memory requirements.

The goal is to understand when an abstraction provides real value and when it simply adds complexity.

> **Good AI engineering is not about adding more components. It is about putting the right responsibility in the right component.**

---

# Current Limitations

This is a learning/engineering prototype rather than a production deployment.

Current limitations include:

* Local in-memory structured memory
* Local ChromaDB persistence
* No distributed state management
* No authentication/authorization layer
* No production secrets management
* No transactional consistency between structured memory and semantic index
* Limited evaluation framework
* Local LLM inference
* No production-grade observability
* No human approval service
* No external enterprise system integrations

These are intentional boundaries for the current stage of the project.

---

# Roadmap

Planned areas for further exploration:

* [ ] Conversation memory
* [ ] Memory extraction integrated with `MemoryManager`
* [ ] RAG + memory combined retrieval
* [ ] Improved tool orchestration
* [ ] Agent evaluation
* [ ] Guardrails
* [ ] Structured output reliability
* [ ] Reranking
* [ ] LangGraph comparison
* [ ] MCP integration
* [ ] Human-in-the-loop workflows
* [ ] Observability and tracing
* [ ] Production persistence
* [ ] API layer
* [ ] Web UI
* [ ] Deployment

---

# Why I Built This

Enterprise AI systems are more than an LLM connected to a vector database.

They require careful separation between:

```text
Language Understanding
        +
Retrieval
        +
Memory
        +
Tool Use
        +
Deterministic Business Logic
        +
Validation
        +
Orchestration
        +
Evaluation
```

This project is an effort to understand and implement those building blocks from first principles, using a realistic enterprise certificate-management scenario.

The emphasis is on **engineering the system around the model**, rather than treating the LLM itself as the entire application.

---

## Author

**Pradeep Reddy Juturu**

Lead Software Engineer | AI Engineering | Full-Stack | Software Architecture

* Portfolio: https://pradeepreddyj-portfolio.vercel.app/
* GitHub: https://github.com/Pradeep-1612
* LinkedIn: https://www.linkedin.com/in/pradeep-reddy-juturu/
