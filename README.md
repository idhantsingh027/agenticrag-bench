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

AgenticRAG-Bench is a research benchmark that evaluates **agentic RAG systems** not just on whether the final answer is correct, but on **how the agent behaved** to get there — what it retrieved, how it planned, how many tool calls it wasted, and how much it cost in tokens.

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

| Week | Questions | System | Embeddings | k | System Prompt | D1 Accuracy | D2 Retrieval | D3 Planning | D5 Efficiency | Key finding |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | llama3.1:8b | 3 | None | 0/3 = 0% | — | D3: 0.04 | 88% wasted | RAGAS answer_relevancy = 0.927 despite 0% accuracy. Agent entered a 27-step retrieval loop → exposes evaluation blind spot. |
| 2a | 5 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | 3 | None | 1/5 = 20% | — | — | 0.763 | Real per-question KB + stronger embeddings eliminate loops, but reveal early stopping and weak multi-hop planning. |
| 2b | 5 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | 5 | None | 0/5 = 0% | — | — | 0.692 | Increasing k does not improve accuracy. Failure is planning, not recall. |
| 2c | 5 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | 5 | Multi-step prompt | 2/5 = 40% | — | — | 0.809 | System prompt enforcement improves accuracy. Remaining errors: entity drift and KB gaps, not retrieval. |
| 3 | 50 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | 3 | Multi-step prompt | 11/50 = 22% | 0.183 | 0.626 | 0.811 | D2 and D3 implemented as metric classes. D3 undershooting fixed — avg steps 1.38 → 2.02. Dominant failure mode shifts from "gave up early" (Type B) to "planned well, retrieved nothing" (Type C, 14/50 questions). Degenerate output detection added to D1. D5 degenerate penalty fixed. |
| 4 | 50 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text | 3 | Multi-step prompt + query rewriting | 15/50 = 30% | 0.302 | 0.647 | 0.860 | Query rewriting → D2 +65% (0.183→0.302), accuracy +36% (22%→30%). D4 implemented: interference rate = −0.133 (noise helped, not hurt). Cross-question swap noise too weak — constructive interference. 8 questions now achieve multi-hit retrieval (D2>0.467), impossible in Week 3. |
| 5 | 50 (MuSiQue) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text + BM25 | 3 | Multi-step prompt + query rewriting + rewrite validation | 24/50 = 48% | 0.416 | 0.698 | 0.883 | Hybrid BM25+FAISS with reciprocal rank fusion. D2=0 cut from 15→7. 13 newly correct, 4 regressions. 39/101 rewrites rejected by validation. D2→D1 pipeline confirmed: better retrieval drives accuracy. |
| 6 | 250 (MuSiQue, 5 conditions) | LangGraph ReAct + Llama 3.1 8B | nomic-embed-text + BM25 | 3 | Multi-step prompt + query rewriting + rewrite validation | A=74%, B=48%, C=64%, D=30%, E=24% | A=0.414, E=0.278 | A=0.599, E=0.494 | — | D6 Difficulty Interaction Collapse confirmed: interaction effect = +14pp (predicted additive 38%, actual 24%). Super-linear collapse — RC costs 26pp, RD costs 10pp, but combined costs 50pp. 8 "interaction victims" pass B and C individually but fail E. |

---

## Week 3 deep-dive

### What changed

Week 3 introduced D2 (Retrieval Step Quality) and D3 (Planning Coherence) as proper metric classes, expanded the benchmark to 50 questions, and fixed three code issues carried over from Week 2.

**Fixes applied:**
- `tokens_per_correct_answer` now stores `null` for incorrect questions instead of `Infinity`
- `_is_degenerate()` added to D1 — catches raw JSON tool dumps and refusal outputs, forces D1=0.0
- D5 now receives the degenerate flag from D1 and returns `efficiency_score: 0.0` instead of a misleading 1.0

### What the scores mean

| Metric | Score | What it tells you |
|---|---|---|
| D1 = 22% | 11/50 correct | 78% of final answers are wrong. D1 is the outcome — it can't say why. |
| D2 = 0.183 | avg retrieval precision | 26/50 questions retrieved zero relevant docs across all search steps. FAISS + nomic-embed-text struggles against MuSiQue's high-distractor paragraphs. |
| D3 = 0.626 | avg planning coherence | Almost everyone makes 2 diverse searches now (up from 1.38 steps avg). Planning is no longer the bottleneck — retrieval is. |
| D5 = 0.811 | avg trajectory efficiency | Clean, non-looping paths. Real cost: 524.2 tokens per correct answer across the 11 successes. |

### Failure pattern breakdown (50 questions)

| Pattern | Count | Description |
|---|---|---|
| Type A: D2=0 + D3=0 + wrong | 2 | Degenerate outputs (Q45, Q47) — agent produced no search steps at all |
| Type B: high D2 + D3=0 + wrong | 0 | Eliminated. Previously 10 questions stopped after one search. System prompt fix closed this mode entirely. |
| Type C: D3≥0.6 + D2=0 + wrong | 14 | **Dominant failure.** Agent plans correctly (2 diverse searches) but FAISS returns nothing relevant both times. |
| Type D: D2≥0.3 + D3≥0.5 + correct | 9 | Ideal path — good retrieval, good planning, correct answer. Up from 4 in the previous run. |

### D2 vs D3 diagnosis

D2 and D3 catch orthogonal failure modes:

- **D2 correct avg = 0.351 vs incorrect = 0.136** (2.5× gap) — retrieval quality is the strongest predictor of correctness
- **D3 correct avg = 0.687 vs incorrect = 0.609** (1.13× gap) — planning coherence gap has narrowed because almost everyone makes 2 searches now

The problem has been correctly re-diagnosed: fixing undershooting (D3) did not fix retrieval (D2). The next lever is query formulation and embedding quality.

---

## Week 4 deep-dive

### What changed

Week 4 introduced two new capabilities: **query rewriting** (LLM rewrites search queries before FAISS) and **D4 noise robustness** (cross-question paragraph swap with 25% distractor replacement).

### Query rewriting effect

Before FAISS search, the agent's raw query is passed to Llama 3.1 8B with the prompt: *"Rewrite this search query to be effective for a vector similarity search. Extract key entities and facts. Remove filler words."* The rewritten query has higher entity density, which produces better vector similarity.

| Metric | Week 3 (no rewriting) | Week 4 (rewriting ON) | Change |
|---|---|---|---|
| D1 Accuracy | 11/50 = 22% | 15/50 = 30% | +36% relative |
| D2 Retrieval | 0.183 | 0.302 | +65% relative |
| D2=0 questions | 24/48 | 15/48 | −37.5% |
| D2>0.467 questions | 0/48 | 8/48 | New capability |
| D3 Planning | 0.626 | 0.647 | Unchanged |

**Key insight:** 8 questions now achieve D2>0.467, meaning the agent retrieves 2+ relevant documents per search step. This never happened in Week 3 (max was 1 relevant doc per step = D2=0.467). Query rewriting doesn't just find *any* relevant doc — it finds *more* relevant docs per query.

### D4 noise robustness — the constructive interference finding

| Metric | Value |
|---|---|
| Correct (clean) | 15/50 |
| Correct (noisy) | 17/50 |
| Interference rate | −0.133 |
| Flipped (right→wrong) | 4 |
| Gained (wrong→right) | 6 |

The interference rate is **negative** — noise helped more than it hurt. This is because:
- 4 of 6 "gained" questions had D2=0 on clean data but D2>0 on noisy data. The injected supporting paragraphs from other questions were higher-quality text than the random distractors they replaced.
- 2 of 4 "flipped" questions had identical D2 in both runs — noise confused the LLM's *reasoning*, not its retrieval.
- The agent is so retrieval-starved (D2=0.302) that any additional relevant-looking text helps more than it confuses.

**Implication:** Cross-question paragraph swap is not adversarial enough. For a real robustness test, need LLM-generated paragraphs with wrong facts about the same entities.

### What the scores mean now

| Metric | Score | What it tells you |
|---|---|---|
| D1 = 30% | 15/50 correct | Query rewriting improved accuracy 22%→30%. Still 70% wrong, but the D2→D1 pipeline is confirmed. |
| D2 = 0.302 | avg retrieval precision | D2=0 dropped from 24 to 15 questions. 8 questions now get multi-hit retrieval. Still the main bottleneck. |
| D3 = 0.647 | avg planning coherence | Flat vs Week 3. Planning is not the differentiator — D3 for correct (0.655) ≈ incorrect (0.643). |
| D4 = −0.133 | interference rate | Noise accidentally helped. Constructive interference reveals retrieval starvation. |
| D5 = 0.860 | avg trajectory efficiency | Slightly improved — query rewriting uses marginally more tokens but improves outcomes. |

---

## Week 6 deep-dive

### What changed

Week 6 implemented D6 (Difficulty Interaction Collapse) — a controlled experiment testing whether accuracy degrades **super-linearly** when multiple difficulty axes are high simultaneously. Instead of finding new questions, we created 5 controlled conditions from the same 50 MuSiQue questions by varying reasoning complexity (RC: 1-hop vs 2-hop) and retrieval difficulty (RD: 6, 12, or 18 distractors).

**1-hop variant creation:** Each 2-hop question has a decomposition (e.g., `"UHF >> distributed by" → "Orion Pictures"` + `"#1 >> founded by" → "Mike Medavoy"`). We take step 2, substitute step 1's answer, and generate a natural-language question: `"Who founded Orion Pictures?"` using 44 relation templates covering all 50 questions.

### The D6 interaction effect — super-linear collapse confirmed

| Condition | RC | Distractors | Accuracy | D2=0 count |
|---|---|---|---|---|
| A (easy baseline) | 1 | 6 | **74%** | 3 |
| C (hard retrieval) | 1 | 18 | **64%** | 10 |
| B (hard reasoning) | 2 | 6 | **48%** | 15 |
| D (medium both) | 2 | 12 | **30%** | 17 |
| E (hard both) | 2 | 18 | **24%** | 20 |

```
Interaction effect calculation:
  RC penalty alone:    74% → 48% = −26pp
  RD penalty alone:    74% → 64% = −10pp
  Predicted additive:  74% − 26% − 10% = 38%
  Actual (E):          24%
  Interaction effect:  38% − 24% = +14pp  ← SUPER-LINEAR
```

If RC and RD were independent, Condition E should score ~38%. It actually scores **24%** — an extra 14pp accuracy loss from the interaction. The difficulties compound.

### Key finding — reasoning is 2.6× harder than retrieval

- **RC penalty** (making questions 2-hop): −26pp
- **RD penalty** (tripling distractors): −10pp

Multi-hop reasoning is the dominant bottleneck for `llama3.1:8b`. Retrieval noise (more distractors) matters, but much less. This aligns with Week 4's D4 finding — the agent is more limited by reasoning than by retrieval quality.

### Per-question patterns

| Pattern | Count | What it means |
|---|---|---|
| A=✓ B=✗ C=✓ E=✗ | 14 | RC-sensitive: 1-hop solves it, 2-hop breaks it regardless of retrieval |
| A=✓ B=✓ C=✓ E=✓ | 9 | Robust core: survives even the hardest condition |
| A=✓ B=✓ C=✓ E=✗ | 7 | **Interaction victims**: handles each axis alone, fails when both hit |
| A=✗ B=✗ C=✗ E=✗ | 7 | Universally hard: can't solve even as 1-hop with 6 distractors |

The 7 "interaction victims" (pass B individually, pass C individually, fail E) are the clearest evidence of super-linear collapse — neither difficulty alone is enough to break the agent, but together they do.

### Confound: tool-calling flakiness

14/50 questions consistently produce empty trajectories in all 2-hop conditions (B, D, E) — the LLM outputs text instead of making a tool call. These same 14 questions work fine as 1-hop (conditions A, C). This adds systematic noise to the 2-hop accuracy numbers but doesn't invalidate the interaction effect, since the 14 empty-trajectory questions are constant across B, D, and E.



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

Open the notebook for the week you want to run.

---

## Project structure

```
agenticrag-bench/
├── notebooks/
│   ├── 1_agenticrag_bench.ipynb     ← Week 1: evaluation gap proof-of-concept
│   ├── 2_agenticrag_bench.ipynb     ← Week 2: D1 + D5 metric classes, 3-way ablation
│   ├── 3_agenticrag_bench.ipynb     ← Week 3: D2 + D3 metrics, 50-question benchmark
│   ├── 4_agenticrag_bench.ipynb     ← Week 4: query rewriting + D4 noise robustness
│   ├── 5_agenticrag_bench.ipynb     ← Week 5: hybrid BM25+FAISS + rewrite validation
│   └── 6_agenticrag_bench.ipynb     ← Week 6: D6 difficulty interaction collapse
├── data/
│   ├── questions/
│   │   └── musique_10.json           ← MuSiQue questions
│   └── trajectories/
│       ├── 1_agent_results.json      ← Week 1 trajectories
│       ├── 1_ragas_scores.json       ← Week 1 RAGAS scores
│       ├── 2_agent_results.json      ← Week 2 trajectories (3 ablation runs)
│       └── 3_agent_results.json      ← Week 3 trajectories (50 questions, D1–D3+D5)
├── notes/
│   ├── 1_observations.md
│   ├── 2_observations.md
│   ├── 3_observations.md
│   ├── 4_observations.md
│   ├── 5_observations.md
│   └── 6_observations.md
├── src/                              ← evaluation harness
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Week 1 — experimental design

Week 1 is not about achieving high MuSiQue accuracy. It is about **demonstrating the evaluation gap**.

### What we did

- Loaded 10 questions from MuSiQue (multi-hop QA dataset)
- Built a small, intentionally limited 15-document knowledge base
- Ran a LangGraph ReAct agent with Llama 3.1 8B via local Ollama
- Logged every tool call, query, retrieved document, token count, and latency
- Ran RAGAS evaluation using Groq (Llama 3.3 70B as judge)
- Compared RAGAS scores against trajectory statistics

### Why the toy knowledge base is intentionally weak

The 15-document KB contains facts about The Godfather, Churchill, and Nobel Prize physicists — not the topics MuSiQue asks about. This creates a **controlled failure mode** where retrieval consistently returns irrelevant documents, the agent loops, and RAGAS answer_relevancy still scores high because hallucinated answers are fluent and on-topic.

### Week 1 finding — the retrieval fixation loop

Question 1 triggered **retrieval fixation** — the agent searched the exact same query `"Green performer spouse"` 27 consecutive times, retrieved the same 3 irrelevant documents each time, and never adapted.

```
Step  1: "Green performer spouse" → [Churchill, Godfather, Friedkin]
Step  2: "Green performer spouse" → [Churchill, Godfather, Friedkin]
...
Step 27: "Green performer spouse" → [Churchill, Godfather, Friedkin]
```

RAGAS cannot detect this. AgenticRAG-Bench D3 flags it immediately.

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

The KB had no documents about the questions being asked. The agent retrieved Churchill and Godfather documents, then answered from Llama's own training memory. Faithfulness correctly scores this as 0.0 — the answer has no grounding in what was retrieved.

</details>

<details>
<summary><strong>Degenerate output (Week 3)</strong></summary>

A degenerate output is when the agent produces a structurally broken answer that has nothing to do with the question — for example, outputting a raw JSON tool-call string (`{"name": "search_knowledge_base"...}`) or a refusal ("Sorry, need more steps to process this request."). These are caught by `_is_degenerate()` in D1 and forced to D1=0.0 and D5=0.0 so they don't pollute averages.

</details>

---

## RAGAS setup (two models, two jobs)

| Role | Model | Where it runs | Why |
|---|---|---|---|
| Agent (answering) | `llama3.1:8b` | Local via Ollama | Free, private, no API needed |
| Embeddings (retrieval) | `nomic-embed-text` | Local via Ollama | Dedicated retrieval model; better similarity quality than llama3.1:8b embeddings |
| RAGAS judge (evaluation) | `llama-3.3-70b-versatile` | Groq (free API) | Stronger model for reliable LLM-as-judge scoring |

---

## Roadmap

- [x] **Week 1** — Core experiment: trajectory logging, RAGAS comparison, retrieval fixation loop documented
- [x] **Week 2** — Evaluation harness: D1 + D5 metric classes, per-question FAISS indexes, loop detection, 3-way ablation (k=3, k=5, k=5+prompt)
- [x] **Week 3** — D2 + D3 metric classes, 50-question benchmark, degenerate output detection, D5 degenerate fix, `tokens_per_correct_answer` fix, undershooting fixed (avg steps 1.38 → 2.02)
- [x] **Week 4** — D4 noise robustness: query rewriting for D2 improvement, cross-question noise injection, interference rate measurement
- [x] **Week 5** — Hybrid retrieval: BM25+FAISS with reciprocal rank fusion, rewrite validation for bad LLM rewrites, D2=0 cut from 15→7, accuracy 30%→48%
- [x] **Week 6** — D6 difficulty interaction collapse: 5 controlled conditions (RC × RD), 250 questions, super-linear collapse confirmed (+14pp interaction effect), 1-hop variants via decomposition, publication-quality plots
- [ ] **Week 7–9** — Full benchmark runs across 4–5 systems and 3+ models
- [ ] **Week 10** — arXiv preprint + public leaderboard on HuggingFace Spaces

---

## Motivating papers

| # | Paper | Key contribution to this project |
|---|---|---|
| 1 | Singh et al. — *Agentic RAG Survey* (arXiv:2501.09136) | Taxonomy of Agentic RAG architectures. Explicitly lists evaluation as an open problem. Primary gap citation. |
| 2 | Lewis et al. — *RAG for Knowledge-Intensive NLP* (NeurIPS 2020) | Original RAG paper. Baseline our benchmark builds beyond. |
| 3 | Lin et al. — *RAGCap-Bench* (arXiv:2510.13910) | Component-level evaluation. Motivates D2 and D3. Gap: 255 questions, no latency/cost, no multi-agent. |
| 4 | Xi et al. — *InfoDeepSeek* (arXiv:2505.15872) | Dynamic web retrieval eval. Discovered retrieval interference. Motivates D4. Gap: no step attribution, no cost tracking. |
| 5 | Narita et al. — *Overcoming RAG Impracticality* (arXiv:2604.02640) | 4-axis difficulty taxonomy. Motivates D6. Gap: 100 questions, single system, no cross-framework. |
