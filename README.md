  # Agentic AI Lab — Technical Documentation

  This repository is a collection of **incremental experiments** for studying agentic AI systems using **LangGraph**, with integrations into LangChain, LangSmith, and external LLM APIs. The focus is not on production readiness but on developing a **rigorous conceptual understanding** of agent orchestration, schema design, and structured evaluation.

  Each notebook explores a narrowly scoped question about agent design: representation of state, control flow, integration with LLMs, and chaining tools into workflows.

  ---

  ## Repository Structure

```
1-LangGraph_Basics/
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

5-RAGs/
├── 1-AgenticRAG.ipynb
├── 2-CoTRAG.ipynb
├── 3-SelfReflection.ipynb
├── 4-QueryPlanningdecomposition.ipynb
├── 5-IterativeRetrieval.ipynb
├── 6-AnswerSynthesis.ipynb
├── 7-MultiAgent.ipynb
├── 8-CorrectiveRAG.ipynb
├── 9-AdaptiveRAG.ipynb
└── 10-RAGWithPersistentMemory.ipynb

```


  * `1-LangGraph_Basics/` → introductory and progressively advanced experiments with LangGraph and its integration with LangChain.
  * `2-LanggraphAdvance/` → advanced streaming and asynchronous execution examples.
  * `3-Debugging/` → reusable agent scripts and tool-augmented graph definitions.
  * `4-Workflows/` → practical workflows demonstrating RAG-enabled ReAct agents with multiple tools.
  * `5-RAGs/` → experiments with agentic RAG workflows using multiple retrievers and LangGraph.
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
  * **State:** defined as a `BaseModel`.

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

  Shows how to bind tools to an LLM (`ChatGroq`) and incorporate them into a **LangGraph pipeline**, handling tool invocation dynamically. Represents a concrete step toward **tool-augmented agents**.

  ---

  ### 7. ReAct Agents (`7-ReActAgents.ipynb`)

  Implements the **ReAct (Reasoning + Acting)** paradigm within LangGraph. Key features include:

  * Combination of reasoning steps with external tool calls.
  * Uses multiple utilities: Arxiv, Wikipedia, Tavily search, and custom math functions (`add`, `sub`, `multi`, `divide`).
  * Integration with **OpenAI models** (`gpt-4o-mini`) for reasoning-driven workflows.
  * Construction of a tool-enabled agent graph with recursive tool use.
  * Introduction of **MemorySaver**, enabling stateful agents that maintain conversational context across interactions.
  * Demonstrates multi-step reasoning workflows and persistent memory via checkpoints.

  ---

  ### 8. ReAct RAG Workflow (`4-Workflows/1-ReAct.ipynb`)

  * Implements a **RAG-enabled ReAct agent** combining reasoning and acting with document retrieval.
  * Integrates tools:

    * **Wikipedia** via `WikipediaQueryRun`.
    * **Arxiv** for academic papers.
    * **Internal tech docs and research notes** via FAISS retrievers.
  * Uses **OpenAI embeddings** for semantic search.
  * **State:** sequence of `BaseMessage` objects for conversational context.
  * **Graph Construction:** `StateGraph` structures multi-tool workflows.
  * **Insight:** Demonstrates **tool-augmented retrieval + reasoning**, enabling agents to answer questions grounded in multiple knowledge bases.

  ---

  ### 9. Agentic RAG Experiment (`5-RAGs/1-AgenticRAG.ipynb`)

  * Implements **agentic RAG** using multiple retrievers (LangGraph + LangChain) and Groq LLM (`ChatGroq`).
  * **Environment:** loads `.env` for `OPENAI_API_KEY` and `GROQ_API_KEY`.
  * **Retrievers:**

    * FAISS-based vector stores built from LangGraph and LangChain tutorials.
    * Tools created via `create_retriever_tool`.
  * **Agent Workflow:**

    * `agent()` → LLM with tool bindings.
    * `grade_documents()` → evaluates relevance of retrieved docs.
    * `generate()` → generates answers from relevant docs.
    * `rewrite()` → reformulates queries if docs not relevant.
  * **State:** `AgentState` (TypedDict with `messages` annotated).
  * **Graph Construction:** `StateGraph` compiles nodes and conditional edges for tool-enabled agentic reasoning.
  * **Visualization:** displays graph using `graph.get_graph(xray=True).draw_mermaid_png()`.
  * **Insight:** Demonstrates end-to-end RAG agent capable of **retrieval, reasoning, query refinement, and multi-tool orchestration**.

  ---

  ### 10. Streaming Responses (`2-LanggraphAdvance/1-streaming.ipynb`)

  * Demonstrates **streaming and asynchronous responses** using LangGraph.
  * **State:** `TypedDict` with annotated `messages` field and `add_messages` handler.
  * **LLM Integration:** OpenAI GPT-4o-mini (`ChatOpenAI`).
  * **Memory:** `MemorySaver` checkpointer for state persistence.
  * **Execution:**

    * `graph_builder.stream()` streams responses in real-time.
    * `graph_builder.astream_events()` asynchronously streams events.
  * **Insight:** Illustrates **real-time interaction** and **threaded conversational context**.

  ---

  ### 11. OpenAI Agent Script (`3-Debugging/openai_agent.py`)

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

  ### 12. CoT RAG Experiment (`5-RAGs/2-CoTRAG.ipynb`)

* Implements **Chain-of-Thought (CoT) reasoning** with a RAG workflow.
* **State:** `CoTState` using Pydantic with fields `question`, `sub_steps`, `retrieved_docs`, and `answer`.
* **Workflow:**
  1. **Planner Node:** Breaks the main question into 2-3 reasoning sub-steps using LLM.
  2. **Retriever Node:** Retrieves documents per sub-step from a FAISS vector store.
  3. **Responder Node:** Synthesizes a well-reasoned final answer using the retrieved documents.
* **LLM Integration:** `ChatGroq` model (`gemma2-9b-it`) used for reasoning and answering.
* **Insight:** Demonstrates **structured multi-step reasoning with retrieval**, combining LangGraph orchestration and document-grounded LLM reasoning.

  ---

### 13. Self-Reflective RAG Agent (`5-RAGs/3-SelfReflection.ipynb`)

Implements a **RAG agent with reflective reasoning**, where the model critiques and improves its own answers in iterative loops.
This experiment studies *self-evaluation* and *revised answer generation* within the LangGraph framework.

* **State:** `RAGReflectionState` (`pydantic.BaseModel`)

  * `question` – input query
  * `retrieved_docs` – retrieved chunks from FAISS store
  * `answer` – generated answer
  * `reflection` – model’s evaluation feedback
  * `revised` – boolean flag to trigger re-generation
  * `attempts` – number of iterations attempted

* **Graph Nodes:**

  * `retriever` → retrieves documents via FAISS.
  * `responder` → generates an answer using the Groq `gemma2-9b-it` model.
  * `reflector` → evaluates the answer and decides whether to revise.
  * `done` → terminates when the answer passes reflection or exceeds 2 attempts.

* **Workflow Logic:**

  1. Retrieve docs relevant to the question.
  2. Generate an answer.
  3. Reflect on correctness and completeness.
  4. If reflection says “NO,” the graph loops back for another iteration.

* **Insight:**
  Demonstrates **autonomous self-correction** — the RAG agent uses reflection to iteratively refine its output, approximating *metacognition* in LLM workflows.

---

### 14. Query Planning and Decomposition RAG (`5-RAGs/4-QueryPlanningdecomposition.ipynb`)

Implements a **Query Planning**–based RAG workflow that decomposes complex questions into simpler sub-queries before retrieval and synthesis.

* **State:** `RAGState` (`pydantic.BaseModel`)

  * `question` – original complex question
  * `sub_question` – list of decomposed sub-queries
  * `retrieved_docs` – documents gathered per sub-question
  * `answer` – final synthesized answer

* **Graph Nodes:**

  * `planner` → breaks complex queries into 2–3 sub-questions using LLM reasoning.
  * `retriever` → retrieves supporting documents for each sub-question.
  * `responder` → generates a unified final answer using retrieved context.

* **Retrievers:**

  * Documents fetched dynamically from URLs such as Lilian Weng’s AI blog posts.
  * Embedded using `OpenAIEmbeddings(model='text-embedding-3-small')` and indexed with FAISS.

* **Insight:**
  This notebook explores **decompositional reasoning**, enabling the agent to answer broad or multi-faceted questions by planning, retrieving, and integrating multiple reasoning chains — key to building **multi-hop reasoning agents**.

  __


### 15. Iterative Retrieval RAG (`5-RAGs/5-IterativeRetrieval.ipynb`)

Implements an **iterative retrieval–generation–reflection loop**, where the agent continuously improves its query and answer quality through self-evaluation.

* **State:** `IterativeRAGState` (`pydantic.BaseModel`)

  * `question`, `refined_question`, `retrieved_docs`, `answer`, `verified`, `attempt`
* **Nodes:**

  * `retrieve_docs` → fetches context via retriever.
  * `generate_answer` → generates initial answer using Groq `gemma2-9b-it`.
  * `reflect_on_answer` → checks if the answer is sufficient.
  * `refine_query` → reformulates the query if reflection says answer is incomplete.
* **Flow:** `retrieve → answer → reflect → refine → retrieve (loop)` until verified or max attempts.
* **Insight:** Models an **autonomous RAG refinement cycle**, achieving self-improving retrieval grounded in feedback loops.

---

### 16. Multi-Source Answer Synthesis (`5-RAGs/6-AnswerSynthesis.ipynb`)

Implements a **multi-source retrieval + synthesis pipeline** that combines information from text documents, YouTube transcripts, Wikipedia, and Arxiv papers.

* **State:** `MultiSourceRAGState` (`pydantic.BaseModel`)

  * `question`, `text_docs`, `yt_docs`, `wiki_context`, `arxiv_context`, `final_answer`
* **Retrievers:**

  * Text retriever (local docs via FAISS)
  * YouTube retriever (manual transcript embedding)
  * Wikipedia via `WikipediaQueryRun`
  * Arxiv using `ArxivLoader`
* **Workflow:**

  * `retrieve_text` → `retrieve_yt` → `retrieve_wiki` → `retrieve_arxiv` → `synthesize`
* **LLM:** Groq `gemma2-9b-it`
* **Insight:** Showcases **cross-source knowledge fusion**, building answers grounded across heterogeneous retrieval sources.

---

### 17. Multi-Agent Collaboration (`5-RAGs/7-MultiAgent.ipynb`)

Explores **collaborative multi-agent systems** with specialized roles — a *Research Agent* and a *Blog Generation Agent* — coordinated through a shared message graph.

* **Core Concept:**
  Agents work *iteratively* through a supervisor, each contributing domain-specific expertise toward a shared task.

* **Architecture:**

  * `research_agent` → gathers and summarizes relevant context using Tavily + internal FAISS retriever.
  * `blog_agent` → synthesizes a structured, human-like blog draft based on the research output.
  * `supervisor_agent` → governs turn-taking between agents, ensuring convergence to “FINAL ANSWER.”

* **Prompt Strategy:**
  Each agent receives a **customized system prompt** clarifying their role, tools, and handoff behavior.

* **Tools Used:**

  * `TavilySearch` (live search)
  * `FAISS retriever` built from local notes
  * LangGraph for orchestration between agents

* **Insight:**
  Demonstrates how *multi-role collaboration* in LangGraph can emulate **hierarchical cognitive workflows**, laying groundwork for **collective intelligence agents**.

---

### 18. Corrective RAG (`5-RAGs/8-CorrectiveRAG.ipynb`)

Implements a **hierarchical self-correcting RAG pipeline** — an autonomous agent that verifies retrieved context and dynamically rewrites its own queries if retrieval quality is low.

* **Core Logic:**

  1. **Retrieve** initial documents from FAISS.
  2. **Grade** their relevance using an LLM-based binary evaluator (`GradeDocuments` schema).
  3. If documents are irrelevant → **transform query** for better search.
  4. **Re-retrieve** or **web search** for external context.
  5. **Generate** a final response using the improved evidence base.

* **Graph Nodes:**

  * `retrieve`
  * `grade_documents`
  * `transform_query`
  * `web_search_node`
  * `generate`

* **Conditional Flow:**
  Uses `StateGraph` with conditional branching:

  ```
  grade_documents → transform_query (if poor relevance)
  grade_documents → generate (if relevance sufficient)
  ```

* **LLMs Used:**
  Groq `gemma2-9b-it` + structured grader model.

* **Prompt Templates:**

  * `grade_prompt` for assessing relevance
  * `rewrite_prompt` for improving queries
  * `rag_prompt` for contextual answer generation

* **Insight:**
  Corrective RAG captures **self-repair** in retrieval systems — the ability to recognize, reason about, and autonomously fix knowledge gaps.

---

Perfect! Let’s extend your **Notebook Summaries** to include the **two new RAG experiments** in the same style, so your Canva-ready README has all 20 notebooks fully documented.

---

### 19. Adaptive RAG (`5-RAGs/9-AdaptiveRAG.ipynb`)

Implements an **adaptive retrieval-augmented generation workflow**, where the agent dynamically chooses retrieval strategies and models based on question type and difficulty.

* **State:** `AdaptiveRAGState` (`pydantic.BaseModel`)

  * `question` – input query
  * `retrieved_docs` – documents fetched
  * `answer` – initial answer
  * `strategy` – retrieval strategy used
  * `confidence` – LLM confidence score for answer

* **Workflow Nodes:**

  * `analyze_question` → classifies question type and decides retrieval strategy
  * `retriever` → fetches documents dynamically from multiple sources (FAISS, Wikipedia, Arxiv)
  * `responder` → generates answer using Groq `gemma2-9b-it` or OpenAI `GPT-4o-mini` depending on strategy
  * `evaluator` → computes confidence; may trigger a **secondary retrieval** if below threshold

* **Insight:** Demonstrates **context-aware, adaptive retrieval and reasoning**, showing how agentic systems can self-optimize based on input complexity.

---

### 20. RAG with Persistent Memory (`5-RAGs/10-RAGWithPersistentMemory.ipynb`)

Combines **RAG workflows with persistent conversational memory**, enabling long-term reasoning across multiple sessions.

* **State:** `PersistentRAGState` (`pydantic.BaseModel`)

  * `conversation_history` – sequence of `BaseMessage` objects across sessions
  * `question` – current user query
  * `retrieved_docs` – documents fetched
  * `answer` – generated answer

* **Workflow Nodes:**

  * `memory_loader` → loads previous conversation history from `MemorySaver`
  * `retriever` → fetches documents based on both current question and historical context
  * `responder` → generates answer grounded in retrieved docs and conversation history
  * `memory_saver` → persists updated conversation for future interactions

* **LLM Integration:** Groq `gemma2-9b-it` or OpenAI `GPT-4o-mini` for reasoning; leverages memory embeddings for context-aware retrieval

* **Insight:** Shows **long-term agentic reasoning**, allowing RAG agents to maintain **contextual awareness and continuity** across multiple interactions — crucial for multi-session, human-like AI assistants.

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
