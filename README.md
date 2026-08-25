# Corporate Learning Path Recommender (Agentic AI + RAG Capstone)

An AI-powered corporate learning path recommendation system built with **LangGraph**, **RAG (FAISS + SentenceTransformers)**, **SQLite LMS Database Integration**, **FastAPI**, and a **Modern Web Frontend (Chat + Visual Timeline)**.
---

GITHUB Link : https://github.com/KaviniMuthamizhvalavan/Trailblazers_Mini_Capstone
---

## 🏗️ Architecture

```
[User message] 
     │
     ▼
[intake_node] ── extracts/updates: goals, background, time availability, constraints list
     │
     ├───────────────────────────┬───────────────────────────┐
     ▼                           ▼                           │
[rag_node]                [lms_tool_node]                    │ (if refinement turn with existing state)
Retrieves course +        Queries SQLite for                 │
prerequisite content       completed & enrolled courses      │
     │                           │                           │
     └───────────────────────────┴───────────────────────────┤
                                 ▼                           │
                       [composition_node] ◄──────────────────┘
                 Merges RAG content + LMS history + ALL
                 accumulated constraints into a sequenced,
                 time-boxed learning path (topological sort)
                                 │
                                 ▼
                 [Returned to Frontend & Saved to Session]
```

### State Management & Multi-Turn Refinement
- **Persistent State (`AgentState`)**: Holds running list of ALL constraints across turns, retrieved RAG context chunks, LMS history, and current path.
- **Direct Refinement Routing**: On follow-up turns, the graph routes directly back into `composition_node` without re-fetching RAG/LMS data, preserving earlier constraints alongside newly introduced ones.

### 🛠️ Native LLM Tool Calling & Graph Routing
- **Native Tool Calling Integration**: When `USE_REAL_TOOL_CALLING` is set to `true` (and configured with a tool-calling capable API key in `NEW_OPENAI_API_KEY`), the graph transitions from a deterministic step to dynamic, LLM-driven tool calling.
- **Agent Node (`lms_agent`)**: The LLM is bound with the `fetch_learner_history` tool via LangChain's `@tool` decorator and `bind_tools()` mechanism. Based on the conversation context and student profile, the LLM dynamically decides whether to execute the database lookup.
- **Prebuilt ToolNode & Extractor**: If the LLM generates a tool call, LangGraph's prebuilt `ToolNode` executes the query, followed by a state extractor node (`lms_tool_extractor_node`) which parses the returned data and updates the graph state keys (`lms_completed`, `lms_enrolled`) for the downstream composition node.
- **Fallback Switch**: If `USE_REAL_TOOL_CALLING` is `false`, the agent transparently falls back to the deterministic parallel pipeline node (`lms_tool_node`), ensuring complete robustness across different model gateways.

---

## 📊 Evaluation & Empirical Results

All metrics below are computed directly from execution against the codebase.

### 1. RAG Retrieval Quality (25% weight)
Evaluated using the actual Ragas library (`answer_relevancy` and `context_recall`) against 3 representative Q&A pairs — a fast-mode subset of the 12-pair test set defined in `eval/ragas_eval.py`:

| Metric | Measured Score | Standard |
|---|---|---|
| **Answer Relevancy** | **96.0% (0.960)** | > 85% |
| **Context Recall** | **100% (1.000)** | > 90% |

Per-question breakdown (from `eval/rag_eval_results.json`):

| Question | Answer Relevancy | Context Recall |
|---|---|---|
| Prerequisites for the Advanced SAP FICO course | 1.000 | 1.000 |
| Courses before the SAP Consultant Certification Prep | 0.896 | 1.000 |
| Beginner courses for cloud computing | 0.983 | 1.000 |

---

### 2. Context-Aware Refinement (35% weight)
Validated across multi-turn conversation scripts (`eval/test_conversations.py`), with full per-turn state captured in `eval/conversation_transcripts.json`. All per-turn assertions pass:

1. **Turn 1 (Goal)**: Learner states "wants Salesforce admin" → Agent recommends the 6-course path `['SF-101', 'SF-102', 'SF-103', 'SF-104', 'SF-105', 'SF-106']` starting with foundational `SF-101`.
2. **Turn 2 (First Constraint)**: Learner adds "no prior experience" → `"no prior Salesforce experience"` is stored as an active constraint; path retains its foundational start `SF-101`.
3. **Turn 3 (Second Constraint)**: Learner adds "only 5 hours a week" → `time_availability` is captured as `"5 hours per week"` **while the Turn 2 constraint remains active** — the multi-constraint-survival test. (The composed course list itself is unchanged from Turn 2; the new constraint is tracked in the dedicated `time_availability` field rather than altering course selection.)
4. **Turn 4 (Contradiction/Revision)**: Learner states "actually I have Salesforce basics" → the `"no prior Salesforce experience"` constraint is marked `superseded: True` and the active-constraint list empties, **while `time_availability` correctly persists as `"5 hours per week"`** — the contradiction affected only the experience claim, not the unrelated time constraint.

---

### 3. Recommendation Relevance (40% weight)
Validated against 5 expert reference personas (`eval/reference_paths.json`) by running the live agent and comparing its composed path against each persona's expert `expected_path`. Aggregate and per-persona results are saved in `eval/relevance_results.json`:

- **Average Overlap Score**: **92.0% (0.920)** across all 5 reference personas (overlap = `|actual ∩ expected| / |expected|`).

| Persona | Overlap | Ordering |
|---|---|---|
| RP-001 (SAP consultant, no background) | 1.00 | 0 violations |
| RP-002 (AWS Solutions Architect) | 1.00 | 0 violations |
| RP-003 (Salesforce admin) | 1.00 | 0 violations |
| RP-004 (AI/ML engineer) | 1.00 | 0 violations |
| RP-005 (Cybersecurity analyst) | 0.60 | 0 violations |

- **Prerequisite Ordering**: Code-enforced via topological sort in `composition_node.py` — guarantees no course appears before its prerequisites (**0 ordering violations across all 5 personas**). For example, `SAP-104` (Advanced FICO) is never scheduled before `SAP-102` (FI) and `SAP-103` (CO).
- **LMS Filtering**: Learner `L001` (Priya Sharma) has completed `SAP-101`, `SAP-102`, `SAP-103`. When requesting an SAP consultant path, the agent recommends `['SAP-106', 'SAP-108']` (excluding the completed courses and the enrolled `SAP-104`) — confirmed in `eval/conversation_transcripts.json`.

---

## 📁 Data Sources

- **Course Catalog (`data/course_catalog.json`)**: 50 synthetic courses across 6 tracks (SAP, Workday, Salesforce, Cloud, AI/ML, Cybersecurity). Features deep prerequisite chains (up to depth 4) and multi-prerequisite nodes to test topological ordering.
- **LMS Records (`data/learner_records.json`)**: 5 learner profiles (`L001` - `L005`) with varied completion histories (beginner, completed prerequisites, cross-domain AI+Cloud).

---

## 🚀 Setup & Execution

### 1. Installation & Environment Setup
```bash
pip install -r requirements.txt --prefer-binary
```

```env
OPENAI_API_KEY= 
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=sqlite:///./db/learning_path.db
```

### 2. Seed Database & Build Vector Store
```bash
python db/seed_db.py
python data/build_vector_store.py
```

### 3. Run Automated Tests
```bash
pytest
```
*All 26 automated unit tests pass in ~2 seconds.*

### 4. Run Evaluation Scripts
```bash
python eval/test_conversations.py
python eval/ragas_eval.py
```

### 5. Start the Web Server
```bash
uvicorn main:app --reload
```
Open browser at: `http://localhost:8000`

---

## 💻 Tech Stack

- **Framework**: Python 3.10, FastAPI, Uvicorn
- **Agent Orchestration**: LangGraph, LangChain
- **Vector Search**: FAISS (`faiss-cpu`), SentenceTransformers (`all-MiniLM-L6-v2`)
- **Database**: SQLite, SQLAlchemy ORM
- **Frontend**: Vanilla HTML5, CSS3 (Dark Mode, Glassmorphism, Micro-animations), JavaScript
