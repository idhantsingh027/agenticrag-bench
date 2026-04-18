<p align="center">
  <h1 align="center">AgenticRAG-Bench</h1>
</p>
<p align="center">
  <em>A multi-dimensional benchmark for evaluating agentic RAG systems</em>
</p>
<p align="center">
  <img src="https://img.shields.io/github/last-commit/idhantsingh027/agenticrag-bench?style=flat&color=007ec6" alt="Last Commit" />
  <img src="https://img.shields.io/github/languages/top/idhantsingh027/agenticrag-bench?style=flat&color=007ec6&cache=none" alt="Top Language" />
</p>
<p align="center">
  <em>Built with the tools and technologies:</em>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/HuggingFace-FF9A00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace" />
</p>

---

## What is this?

AgenticRAG-Bench is a research benchmark that evaluates **agentic RAG systems** not just on whether the final answer is correct, but on **how the agent behaved** to get there — what it retrieved, how it planned, how many tool calls it wasted, and how much it cost in tokens and latency.

> **Core thesis:** RAGAS-style evaluation scores the final answer. It cannot see that an agent searched the same query 27 times, wasted 88% of its retrieval calls, or achieved 0.927 answer relevancy while getting 0% of answers actually correct. AgenticRAG-Bench measures both the outcome and the process.

---

## The problem in one table

| What happened (Week 1 experiment) | RAGAS score | AgenticRAG-Bench score |
|---|---|---|
| Agent searched same query 27 times in a row | Not measured | D3 planning coherence: 0.04 |
| 88% of retrieval steps were redundant | Not measured | D5 trajectory efficiency: 0.12 |
| Answer relevancy 0.93, actual accuracy 0% | answer_relevancy: 0.927 | D1 correctness: 0.000 |
| 3,429 tokens used for one wrong answer | Not measured | D5 tokens/correct-answer: ∞ |

---

## Evaluation dimensions (D1–D6)

| Dimension | What it measures | Paper motivation |
|---|---|---|
| **D1** Answer correctness | Is the final answer right? | Lewis et al. 2020 (original RAG) |
| **D2** Retrieval step quality | Is each individual retrieval call relevant? | RAGCap-Bench (Lin et al. 2025) |
| **D3** Planning coherence | Does the query sequence logically narrow toward the answer? | RAGCap-Bench + Enterprise RAG |
| **D4** Noise robustness | Does accuracy drop when conflicting sources are present? | InfoDeepSeek (Xi et al. 2025) |
| **D5** Trajectory efficiency | How many tokens and steps per correct answer? | Singh et al. survey 2025 |
| **D6** Difficulty interaction collapse | Does performance degrade super-linearly when multiple difficulty axes are high simultaneously? | Enterprise RAG (Narita et al. 2026) |

---

## Results so far

| Week | Questions | System | Embeddings | Retrieval k | System Prompt | D1 Accuracy | D5 Efficiency | Key finding |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | llama3.1:8b | k=3 | None | 0/3 = 0% | 88% wasted | RAGAS answer_relevancy = 0.927 despite 0% accuracy. Agent entered a 27-step retrieval loop → exposes evaluation blind spot. |
| 2a | 5 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | k=3 | None | 1/5 = 20% | 0.763 | Real KB + stronger embeddings eliminate loops, but reveal early stopping and weak multi-hop planning. |
| 2b | 5 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | k=5 | None | 0/5 = 0% | 0.692 | Increasing k does not improve performance. Agent still performs single-step retrieval → failure is not recall but planning. |
| 2c | 5 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | k=5 | Multi-step prompt + tool constraints | 2/5 = 40% | 0.809 | Enforcing multi-step behavior improves accuracy. Remaining errors stem from reasoning instability (entity drift) and KB gaps, not retrieval. |

---

## Quickstart

### Prerequisites

```bash
# 1. Ollama installed and running
ollama serve

# 2. Pull the model
ollama pull llama3.1:8b

# 3. Python 3.11+ required
python3.11 --version
```

### Setup

```bash
git clone https://github.com/idhantsingh027/agenticrag-bench.git
cd agenticrag-bench

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### Run

```bash
# Make sure Ollama is running in a separate terminal first
jupyter notebook
```

Open `notebooks/1_agenticrag_bench.ipynb` and run all cells.

---

## Project structure

```
agenticrag-bench/
├── notebooks/
│   └── 1_agenticrag_bench.ipynb      ← Week 1 experiment
├── data/
│   ├── questions/
│   │   └── musique_10.json           ← 10 MuSiQue questions
│   └── trajectories/
│       ├── 1_agent_results.json      ← agent trajectories
│       └── 1_ragas_scores.json       ← RAGAS scores
├── notes/
│   └── week1_observations.md         ← research observations
├── src/                              ← evaluation harness (Week 2+)
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Week 1 — experimental design

Week 1 is not about achieving high MuSiQue accuracy. It is about **demonstrating the evaluation gap**.

### What we did

- Loaded 10 questions from MuSiQue (multi-hop QA dataset)
- Built a small intentionally limited 15-document knowledge base
- Ran a LangGraph ReAct agent with Llama 3.1 8B via local Ollama
- Logged every tool call, query, retrieved document, token count, and latency
- Ran RAGAS evaluation using Groq (Llama 3.3 70B as judge) on the same results
- Compared RAGAS scores against trajectory statistics

### Why the toy knowledge base is intentionally weak

The 15-document KB contains facts about The Godfather, Churchill, and Nobel Prize physicists — not the topics MuSiQue asks about. This creates a **controlled failure mode** where:

- Retrieval consistently returns irrelevant documents
- The agent cannot find the answer and loops
- RAGAS answer_relevancy still scores high because the agent's hallucinated answers are fluent and on-topic

This demonstrates that high answer_relevancy does not imply good retrieval behavior or correct answers.

### Week 1 finding — the retrieval fixation loop

Question 1 ("Who is the spouse of the Green performer?") triggered a behavior we call **retrieval fixation** — the agent searched the exact same query `"Green performer spouse"` 27 consecutive times, retrieved the same 3 irrelevant documents each time, and never adapted its strategy.

```
Step  1: "Green performer spouse" → [Churchill, Godfather, Friedkin]
Step  2: "Green performer spouse" → [Churchill, Godfather, Friedkin]
...
Step 27: "Green performer spouse" → [Churchill, Godfather, Friedkin]
```

RAGAS cannot detect this. AgenticRAG-Bench D3 (planning coherence) flags it immediately.

---

## Key terms

<details>
<summary><strong>RAG vs. Agentic RAG</strong></summary>

**RAG:** retrieve documents → generate answer. One retrieval step, fixed pipeline.

**Agentic RAG:** an agent decides whether to retrieve, what query to use, whether to retrieve again, and when to stop. Multiple steps, dynamic decisions.

</details>

<details>
<summary><strong>ReAct agent</strong></summary>

ReAct = Reason + Act. The model alternates between reasoning (thinking about what it needs) and acting (calling a tool like the retrieval function). In this repo the primary action is calling the vector search tool.

</details>

<details>
<summary><strong>Trajectory</strong></summary>

A trajectory is the complete record of what an agent did to answer one question — every query it searched, every document it retrieved, how long each step took, and how many tokens it used. RAGAS ignores the trajectory. AgenticRAG-Bench scores it.

</details>

<details>
<summary><strong>RAGAS metrics</strong></summary>

- **faithfulness** — is the answer supported by the retrieved documents?
- **answer_relevancy** — is the answer on-topic for the question?
- **context_precision** — how much of what was retrieved was actually useful?

All three measure properties of the final answer and final retrieved context. None measure the agent's intermediate decisions.

</details>

<details>
<summary><strong>Why faithfulness was 0.0 in Week 1</strong></summary>

The knowledge base had no documents about the questions being asked. The agent retrieved Churchill and Godfather documents for a question about a musician's spouse. Its answer came from Llama's own training memory, not from the retrieved documents. Faithfulness correctly scores this as 0.0 — the answer has no grounding in what was retrieved.

</details>

---

## RAGAS setup (two models, two jobs)

Week 1 uses two different models for two different purposes:

| Role | Model | Where it runs | Why |
|---|---|---|---|
| Agent (answering) | `llama3.1:8b` | Local via Ollama | Free, private, no API needed |
| Embeddings (retrieval) | `llama3.1:8b` | Local via Ollama | Converts text to vectors for FAISS search |
| RAGAS judge (evaluation) | `llama-3.3-70b-versatile` | Groq (free API) | Stronger model for reliable LLM-as-judge scoring, handles parallel eval calls |

> **Note for Week 2:** Week 1 uses `llama3.1:8b` for both generation and embeddings. Week 2 will switch embeddings to `nomic-embed-text` — a dedicated retrieval model that gives significantly better similarity search quality.

---

## Motivating papers

| # | Paper | Key contribution to this project |
|---|---|---|
| 1 | Singh et al. — *Agentic RAG Survey* (arXiv:2501.09136) | Taxonomy of Agentic RAG architectures. Explicitly lists evaluation as an open problem. Primary gap citation. |
| 2 | Lewis et al. — *RAG for Knowledge-Intensive NLP* (NeurIPS 2020) | Original RAG paper. Baseline our benchmark builds beyond. |
| 3 | Lin et al. — *RAGCap-Bench* (arXiv:2510.13910) | Component-level evaluation. Motivates D2 and D3. Gap: 255 questions, no latency/cost, no multi-agent. |
| 4 | Xi et al. — *InfoDeepSeek* (arXiv:2505.15872) | Dynamic web retrieval eval. Discovered retrieval interference. Motivates D4. Gap: no step attribution, no cost tracking. |
| 5 | Narita et al. — *Overcoming RAG Impracticality* (arXiv:2604.02640) | 4-axis difficulty taxonomy. Motivates D6. Gap: 100 questions, single system, no cross-framework. |

---

## Roadmap

- [x] **Week 1** — Core experiment: trajectory logging, RAGAS comparison, finding documented
- [ ] **Week 2** — Evaluation harness: refactor trajectory logger, implement D1 + D5 as proper metrics, add `max_steps` loop prevention, use MuSiQue supporting paragraphs as knowledge base
- [ ] **Week 3** — Dataset v1: 400+ labelled questions on HuggingFace, 4-axis difficulty tags
- [ ] **Week 4–6** — Implement D2, D3, D4, D6 metrics
- [ ] **Week 7–9** — Full benchmark runs across 4–5 systems and 3+ models
- [ ] **Week 10** — arXiv preprint + public leaderboard on HuggingFace Spaces
