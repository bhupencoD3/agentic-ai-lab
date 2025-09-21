# Agentic AI Lab — Technical Documentation

This repository is a collection of **incremental experiments** for studying agentic AI systems using **LangGraph**, with integrations into LangChain, LangSmith, and external LLM APIs. The focus is not on production readiness but on developing a **rigorous conceptual understanding** of agent orchestration, schema design, and structured evaluation.

Each notebook explores a narrowly scoped question about agent design: representation of state, control flow, integration with LLMs, and chaining tools into workflows.

---

## Repository Structure

```
1-LangGraph_Basics/
│
├── 1-simple_graph.ipynb
├── 2-chatbot.ipynb
├── 3-DataClasses_StateSchema.ipynb
├── 4-pydantic.ipynb
├── 5-ChainsLangGraph.ipynb
├── 6-Chatbot_multiple_tools.ipynb
└── 7-ReActAgents.ipynb

2-LanggraphAdvance/
├── 1-streaming.ipynb

3-Debugging/
├── openai_agent.py

4-Workflows/
├── 1-ReAct.ipynb
```

* `1-LangGraph_Basics/` → introductory and progressively advanced experiments with LangGraph and its integration with LangChain.
* `2-LanggraphAdvance/` → advanced streaming and asynchronous execution examples.
* `3-Debugging/` → reusable agent scripts and tool-augmented graph definitions.
* `4-Workflows/` → practical workflows demonstrating RAG-enabled ReAct agents with multiple tools.
* `environment.yml` → conda environment file for reproducing the experiments.

---

## Notebook Summaries

### 1. Simple Graph (`1-simple_graph.ipynb`)

* Introduces LangGraph as a **state machine**.
* **State:** represented as a `TypedDict` with a `graph_info` field.
* **Nodes:** simple Python functions (`start_play`, `cricket`, `football`) that enrich state.
* **Control Flow:** probabilistic branching using `random.random()`.
* **Insight:** Demonstrates that LangGraph enables **deterministic computation with stochastic routing**, a foundation for probabilistic agent behavior.

---

### 2. Chatbot (`2-chatbot.ipynb`)

* Embeds LLMs as **graph nodes**.
* **State:** holds a message list, managed by LangGraph’s `add_messages`.
* **Backends:** OpenAI GPT-4o-mini (`ChatOpenAI`) and Groq Gemma2-9B (`ChatGroq`).
* **Execution:**

  * `invoke()` with user messages.
  * `stream()` to observe real-time conversational propagation.
* **Insight:** Shows how **symbolic graph structure** integrates with **sub-symbolic LLM reasoning**.

---

### 3. DataClasses and State Schema (`3-DataClasses_StateSchema.ipynb`)

* Explores state schema design trade-offs.
* **TypedDict Schema:** lightweight, dictionary-like, enforces strict typing.
* **Dataclass Schema:** richer semantics, supports object-oriented patterns.
* **Graphs:** both schema styles compile into the same DAG topology.
* **Insight:** State representation is a **theoretical commitment** — it impacts correctness, reproducibility, and extensibility of agent workflows.

---

### 4. Pydantic State Validation (`4-pydantic.ipynb`)

* Introduces **Pydantic models** for stricter runtime validation.
* **State:** defined as a `BaseModel`:

```python
class State(BaseModel):
    name: str
```

* **Node:** returns a modified state (`{"name": "hello"}`).
* **Execution:**

  * Accepts valid string input (`{"name": "str"}`).
  * Rejects invalid input (`{"name": 123}`) with a `ValidationError`.
* **Insight:** Using Pydantic enforces **schema integrity at runtime**, preventing silent failures and supporting robust agent workflows.

---

### 5. Chains in LangGraph (`5-ChainsLangGraph.ipynb`)

* Connects **LangChain message structures** with LangGraph execution.
* **Conversation Representation:** sequences of `AIMessage` and `HumanMessage`.
* **Tool Binding:**

  * Defines a simple tool `add(a, b)`.
  * Binds it to an LLM using `.bind_tools()`.
  * Invokes tool calls dynamically from natural language queries.
* **Graph Construction:**

  * Node `llm_tool` handles tool-augmented LLM responses.
  * Conditional edges (`tools_condition`) route execution to tool nodes when appropriate.
* **Insight:** Demonstrates **tool-augmented reasoning** within LangGraph, a step toward agentic systems capable of **planning + acting** workflows.

---

### 6. Chatbot with Multiple Tools (`6-Chatbot_multiple_tools.ipynb`)

Demonstrates the extension of a chatbot with **multiple external tools**:

* **Arxiv** for academic paper queries.
* **Wikipedia** for encyclopedic knowledge.
* **Tavily** for up-to-date web search.

The notebook shows how to bind tools to an LLM (`ChatGroq`) and incorporate them into a **LangGraph pipeline**. Tool invocation is handled dynamically based on user queries. Represents a concrete step toward **tool-augmented agents**.

---

### 7. ReAct Agents (`7-ReActAgents.ipynb`)

Implements the **ReAct (Reasoning + Acting)** paradigm within LangGraph. Key features include:

* Combination of reasoning steps with external tool calls.
* Use of multiple utilities: Arxiv, Wikipedia, Tavily search, and custom math functions (`add`, `sub`, `multi`, `divide`).
* Integration with **OpenAI models** (`gpt-4o-mini`) for reasoning-driven workflows.
* Construction of a tool-enabled agent graph with recursive tool use.
* Introduction of **MemorySaver**, enabling stateful agents that maintain conversational context across interactions.

The notebook demonstrates advanced scenarios where the agent reasons over multi-step tasks (e.g., searching news, performing arithmetic, and chaining results). Highlights the role of persistent memory via checkpoints for multi-turn, state-aware conversations.

---

### 8. ReAct RAG Workflow (`4-Workflows/1-ReAct.ipynb`)

* Implements a **RAG-enabled ReAct agent** combining reasoning and acting with document retrieval.
* Integrates tools:

  * **Wikipedia** via `WikipediaQueryRun`.
  * **Arxiv** for academic papers.
  * **Internal tech docs and research notes** via FAISS retrievers.
* Uses **OpenAI embeddings** for semantic search.
* **State:** maintained as a sequence of `BaseMessage` objects, enabling conversational context.
* **Graph Construction:** uses `StateGraph` to structure multi-tool workflows.
* **Example Query:** `"what does research says about 'Ethical and Societal Considerations'?"`
* **Insight:** Demonstrates **tool-augmented retrieval + reasoning**, enabling agents to answer questions grounded in external and internal knowledge bases.

---

### 9. Streaming Responses (`2-LanggraphAdvance/1-streaming.ipynb`)

* Demonstrates **streaming and asynchronous responses** using LangGraph.
* **State:** `TypedDict` with annotated `messages` field and `add_messages` handler.
* **LLM Integration:** OpenAI GPT-4o-mini (`ChatOpenAI`).
* **Memory:** `MemorySaver` checkpointer for state persistence.
* **Execution:**

  * `graph_builder.stream()` streams responses in real-time.
  * `graph_builder.astream_events()` asynchronously streams events.
* **Insight:** Illustrates **real-time interaction** and **threaded conversational context**.

---

### 10. OpenAI Agent Script (`3-Debugging/openai_agent.py`)

* Provides reusable **agent graph scripts** for experimentation.
* **State:** `TypedDict` with annotated `messages` field using `BaseMessage`.
* **LLM Integration:** OpenAI GPT-4o-mini (`ChatOpenAI`) with optional temperature control.
* **Graphs:**

  * `make_default_graph()` – basic graph with a single LLM node.
  * `make_alternate_graph()` – graph with tool augmentation and conditional routing.
* **Tools:** Uses `ToolNode` and `@tool` decorators.
* **Execution Flow:** conditional edges check for `tool_calls`.
* **Insight:** Modularizes agent definitions, integrates tools, and enables scalable experimentation.

---

## Broader Research Trajectory

1. **Control Flow:** deterministic vs stochastic routing.
2. **Representation:** schema choices (`TypedDict`, `dataclass`, `pydantic`) shape rigor and maintainability.
3. **Integration:** LLMs as nodes, tools as callable functions, messages as state.
4. **Validation:** runtime enforcement via Pydantic ensures integrity of state transformations.
5. **Agentic Workflows:** chaining LLMs and tools under a graph paradigm approximates **planning–action–evaluation cycles**.
6. **Streaming & Async:** real-time streaming and asynchronous events enable responsive, stateful agent interactions.

---

## Planned Directions

* Variations of **RAG pipelines** integrated into LangGraph.
* Multi-agent coordination workflows.
* Structured evaluation using **LangSmith**.
* Comparative benchmarks across schema strategies.
* Advanced streaming and async interaction patterns for responsive agents.

---

## License

MIT License.

---

## Author

**Bhupen**

* [LinkedIn](https://www.linkedin.com/in/bhupenparmar/)
* [GitHub](https://github.com/bhupencoD3)
