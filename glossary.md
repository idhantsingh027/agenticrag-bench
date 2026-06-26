# AgenticRAG-Bench — Glossary

> This file explains every key term, concept, and variable in the project in simple language with examples. Read this before going through the notebooks.

---

## The one-line summary

We built a way to properly grade an AI that searches documents to answer questions — not just "did it get the answer right?" but "did it search sensibly, find the right things, and not waste effort?"

---

## Part 1 — What is the project about?

### The core problem

When you ask an AI to answer a question using a set of documents, the standard way to grade it (called RAGAS) only looks at the final answer. It asks: "Is the answer on-topic? Is it supported by what was retrieved?"

The problem is that RAGAS gives high scores even when the AI is behaving terribly. In our Week 1 experiment:

- The AI searched the exact same phrase 27 times in a row
- It retrieved the same 3 useless documents every time
- It never found the answer
- RAGAS still gave it a score of **0.927 out of 1.0** (excellent)
- Actual correct answers: **0%**

AgenticRAG-Bench is the evaluation framework we're building to catch all the things RAGAS misses.

### What "agentic" means

A normal RAG (Retrieval-Augmented Generation) system works like this:
```
Question → Search once → Get documents → Generate answer
```

An **agentic** RAG system is smarter — it can decide:
- What to search for
- Whether the results were good enough
- Whether to search again with a different query
- When to stop and give an answer

Think of it like a detective. A normal system grabs the first file it finds and writes a report. An agentic system looks at the first file, decides it needs more evidence, searches again, and builds the answer step by step.

Our benchmark grades the detective's behaviour — not just whether they named the right suspect.

---

## Part 2 — The dataset (MuSiQue)

### What is MuSiQue?

MuSiQue (Multi-hop Questions) is a dataset of questions that **require multiple steps to answer**. You can't look up the answer in one search — you have to find an intermediate fact first.

**Example of a 2-hop question:**
> "What country is the birthplace of the founder of the company that makes the iPhone?"

- **Hop 1:** Who founded the company that makes the iPhone? → Steve Jobs
- **Hop 2:** Where was Steve Jobs born? → San Francisco, USA

A single search for "iPhone country" won't work. You need to chain two searches together.

**Example of a 3-hop question:**
> "What is the capital of the country where the director of Parasite was born?"

- **Hop 1:** Who directed Parasite? → Bong Joon-ho
- **Hop 2:** Where was Bong Joon-ho born? → South Korea
- **Hop 3:** What is the capital of South Korea? → Seoul

Our benchmark runs these questions through an AI agent and grades how well it navigated the hops.

### What are supporting paragraphs?

For each question, MuSiQue provides the specific paragraphs that contain the answer. We use these as our **knowledge base** — the documents the agent is allowed to search through.

We also use them as ground truth for D2: if the agent retrieved a supporting paragraph, that's a good retrieval step. If it retrieved something irrelevant, that's a bad step.

---

## Part 3 — The AI system

### LangGraph ReAct agent

The AI we're testing is a **ReAct agent** built with LangGraph and powered by Llama 3.1 8B running locally on Ollama.

**ReAct** stands for Reason + Act. The agent alternates between two things:
1. **Reasoning:** "I need to find out who directed Parasite first"
2. **Acting:** Calls the search tool with the query "director of Parasite"

Then it looks at the results and reasons again:
1. **Reasoning:** "OK, it's Bong Joon-ho. Now I need his birthplace"
2. **Acting:** Calls the search tool with "Bong Joon-ho birthplace"

And so on until it has enough to answer, or it gives up.

**LangGraph** is the framework that manages this loop — deciding when the agent should search again vs. when it should stop and give the final answer.

**Llama 3.1 8B** is the language model doing the reasoning. "8B" means 8 billion parameters — a mid-size model that runs locally without a GPU. It's capable but not as powerful as GPT-4 or Claude.

### FAISS and nomic-embed-text

**FAISS** (Facebook AI Similarity Search) is the search engine the agent uses. It doesn't search by keywords — it converts text to numbers (vectors) and finds documents that are mathematically similar to the query.

**nomic-embed-text** is the model that does this conversion. It's a dedicated embedding model — its only job is turning text into vectors for search. It's better at this specific task than using the main LLM for embeddings.

**k** is the number of documents FAISS returns per search. If k=3, every search call returns 3 documents. If k=5, it returns 5.

**Why k matters:** A higher k casts a wider net — you might catch a relevant document you'd miss with k=3. But it also means the agent has more documents to read per step, which costs more tokens and can confuse it. Our Week 2 ablation found that increasing k from 3 to 5 didn't help — the failure was in planning, not in how many documents were returned.

---

## Part 4 — The 6 evaluation dimensions (D1–D6)

These are the six ways we grade the agent. Think of them as six different questions a good coach would ask after watching a game.

---

### D1 — Answer Correctness

**Simple version:** Did the agent get the right answer?

**How it's measured:** The agent's final answer is compared to the ground-truth answer using exact match, token overlap, and sequence similarity. A score of 1.0 means correct. 0.0 means completely wrong.

**Example:**
- Ground truth: `"Martin Eberhard"`
- Agent answer: `"Elon Musk"` → D1 = 0.0
- Agent answer: `"Martin Eberhard"` → D1 = 1.0
- Agent answer: `"Martin Eberhard co-founded Tesla"` → D1 partial (contains the answer but isn't an exact string match)

**What D1 cannot tell you:** Whether the agent was close, whether it retrieved the right context, or why it failed. It's purely the outcome.

**Week 3 result:** 11/50 = 22% accuracy. 29 questions scored D1=0.0 (complete miss).

**Degenerate outputs:** A special case where the agent produces broken output — like printing a raw JSON tool call instead of an answer, or saying "Sorry, I can't process this." These are caught by `_is_degenerate()` and forced to D1=0.0 so they don't pollute the averages.

---

### D2 — Retrieval Step Quality

**Simple version:** Each time the agent did a search, did it actually find useful documents?

**How it's measured:** For each search call, we check how many of the k=3 returned documents are actually from the MuSiQue supporting paragraphs (the ones that contain the answer). This is called **precision at k**.

The formula: `d2 = avg_precision × 0.8 + 0.2 (if any relevant doc was ever retrieved)`

**Example with k=3:**
- Search returns: [relevant doc, irrelevant doc, irrelevant doc] → precision = 1/3 = 0.333 → D2 = 0.333×0.8+0.2 = 0.467
- Search returns: [irrelevant, irrelevant, irrelevant] → precision = 0/3 = 0 → D2 = 0.0
- Search returns: [relevant, relevant, irrelevant] → precision = 2/3 = 0.667 → D2 = 0.733

**Why D2 is quantized (only 4 possible values):** Because k=3 means precision can only be 0/3, 1/3, 2/3, or 3/3. Our agent never achieved 2/3 or 3/3 — it never got more than 1 relevant doc per search. The highest D2 you'll see is 0.467.

**What D2 reveals:** In Week 3, 26 out of 50 questions had D2=0.0 across all their search steps — the agent retrieved nothing useful in any of its searches. This points to the embedding model (nomic-embed-text) struggling against MuSiQue's distractor paragraphs, not to the agent being lazy.

**D2 correct vs incorrect:** Questions the agent got right had avg D2=0.351. Questions it got wrong had avg D2=0.136. This is a 2.5× gap — retrieval quality is the strongest predictor of whether the agent gets the answer right.

---

### D3 — Planning Coherence

**Simple version:** Did the agent search in a sensible, progressive way — or did it loop, give up too early, or jump randomly between topics?

**How it's measured:** Three components combined:
- **Hop coverage:** Did the agent make enough searches for the number of hops required? (1 search for a 2-hop question = bad)
- **Query diversity:** Were the queries meaningfully different from each other? (identical queries = looping; completely unrelated queries = random pivoting)
- **Undershoot penalty:** If the agent made fewer searches than hops needed, it takes a penalty of `0.5 × undershoot_fraction`

**Why 1-step questions score D3=0.0:** For a 2-hop question with only 1 unique search:
- hop_coverage = 0.5 (did 1, needed 2)
- diversity = 0.0 (can't measure diversity with only 1 query)
- undershoot_penalty = 0.25

Final: `0.5×0.5 + 0.5×0.0 − 0.25 = 0.0`

This is intentional — making only 1 search on a 2-hop question is definitively bad planning, and D3 calls it zero.

**Example of a good D3 path (for a 2-hop question about a director's birthplace):**
```
Search 1: "director of Parasite"           → returns relevant doc about Bong Joon-ho ✓
Search 2: "Bong Joon-ho birthplace"        → returns relevant doc about South Korea ✓
D3 ≈ 0.80 (2 diverse searches, progressive narrowing)
```

**Example of a bad D3 path:**
```
Search 1: "director of Parasite"           → returns relevant doc about Bong Joon-ho ✓
(agent immediately answers without searching for birthplace)
D3 = 0.0 (only 1 search, severe undershoot penalty)
```

**The Week 2 → Week 3 shift:** In Week 2, the agent averaged 1.38 steps — most questions only got 1 search. After adding system prompt enforcement ("you must search at least twice"), the avg jumped to 2.02. D3 went from 0.297 to 0.626. This fixed the planning problem but didn't fix retrieval — Type C failures (good planning, empty retrieval) became the new dominant failure mode.

---

### D4 — Noise Robustness *(implemented — Week 4)*

**Simple version:** If we plant misleading documents in the knowledge base, does the agent get confused?

**How it's measured:** Run each question twice:
1. Clean run — normal knowledge base
2. Noisy run — 25% of distractor paragraphs replaced with supporting paragraphs from other questions (cross-question swap)

Then compute **interference rate:**
```
interference_rate = (correct_clean − correct_noisy) / correct_clean
```

If interference_rate = 1.0, the agent is completely fooled by noise. If 0.0, it's perfectly robust. If **negative**, noise accidentally *helped* the agent (more on this below).

**The D4 metric class** also tracks per-question detail:
- **flipped:** Questions the agent got right on clean data but wrong on noisy data (noise hurt)
- **gained:** Questions the agent got wrong on clean data but right on noisy data (noise helped)

**Week 4 result — the surprise:**
```
Correct (clean):   15/50
Correct (noisy):   17/50
Interference rate:  −0.133
Flipped (right→wrong): 4
Gained  (wrong→right): 6
```

The interference rate is **negative** — the agent performed *better* on noisy data. This happened because:
- 4 of 6 "gained" questions had D2=0 on clean data (FAISS retrieved nothing useful) but D2>0 on noisy data
- The injected supporting paragraphs from other questions were higher-quality text than the random distractors they replaced, so FAISS found them and the LLM extracted enough context to answer
- The agent is so retrieval-starved (D2=0.302 on clean) that any additional relevant-looking text helps more than it confuses

**What this means:** Cross-question paragraph swap produces **constructive interference** at this difficulty level — the noise isn't adversarial enough. For a real adversarial test, we'd need LLM-generated paragraphs with wrong facts about the *same* entities (e.g., "Tesla was founded in 2002 by Elon Musk" instead of the real founder).

**The key finding from D4 is not "the agent is robust" — it's "the agent is so bad at retrieval that even random helpful noise improves it."** RAGAS has no mechanism to detect this at all.

---

### D5 — Trajectory Efficiency

**Simple version:** Did the agent do the work cleanly, without wasting steps or tokens?

**How it's measured:**
```
efficiency_score = 1.0 − (redundancy × 0.5) − (token_penalty × 0.3) − loop_penalty
```

- **redundancy:** fraction of search queries that were near-identical to a previous one
- **token_penalty:** how many tokens per step, normalised against a baseline of 400 tokens/step
- **loop_penalty:** 0.3 if any exact query repetition was detected

**`tokens_per_correct_answer`:** The total tokens used divided by whether the answer was correct. For incorrect answers, this is `null` (the agent spent tokens and produced nothing useful). The real efficiency number is the average over correct questions only: **524.2 tokens per correct answer** in Week 3.

**The D5 trap:** D5 rewards being brief. An agent that immediately gives up and outputs garbage scores near 1.0 — because it used zero tokens and made zero redundant queries. In Week 3, our two degenerate questions (Q45, Q47) previously scored D5=1.0 (perfect). After the fix, they score D5=0.0. Always read D5 alongside D1 — high D5 + low D1 means the agent was efficiently wrong.

**Week 3 result:** Avg D5=0.811. This dropped from 0.913 because (a) the degenerate fix removed the fake 1.0 scores and (b) all questions now take 2 steps instead of 1, which uses slightly more tokens.

---

### D6 — Difficulty Interaction Collapse *(implemented — Week 6)*

**Simple version:** Does the agent fall apart much faster when a question is hard in multiple ways at once, rather than just hard in one way?

**Background — the 2 difficulty axes tested in Week 6:**
- **Reasoning Complexity (RC):** How many hops? RC=1 (single-hop, easy) vs RC=2 (2-hop, hard).
- **Retrieval Difficulty (RD):** How many distractor documents? RD=1 (2 distractors), RD=2 (8 distractors), RD=3 (18 distractors, original).

**The D6 hypothesis:** A question with RC=2 (hard reasoning) might cause X% accuracy drop. A question with RD=3 (many distractors) might cause Y% accuracy drop. But a question with BOTH RC=2 AND RD=3 might cause more than X%+Y% accuracy drop. This super-linear degradation is what D6 measures.

**How Week 6 tests this — controlled difficulty variants:**
Instead of finding new questions, we create 5 conditions from the *same* 50 questions by manipulating RC and RD:

| Condition | RC | Distractors | Accuracy | Description |
|---|---|---|---|---|
| A | 1 | 6 | **74%** | Easy baseline: 1-hop question, 6 distractors |
| B | 2 | 6 | **48%** | Hard reasoning only: 2-hop, 6 distractors |
| C | 1 | 18 | **64%** | Hard retrieval only: 1-hop, 18 distractors |
| D | 2 | 12 | **30%** | Medium both: 2-hop, 12 distractors |
| E | 2 | 18 | **24%** | Hard both: 2-hop, 18 distractors (original) |

**Creating 1-hop variants (RC=1):** Each 2-hop question has a decomposition like `"UHF >> distributed by" → "Orion Pictures"` + `"#1 >> founded by" → "Mike Medavoy"`. We take step 2 and expand it into a natural-language question: `"Who founded Orion Pictures?"` using relation templates.

**Creating low-distractor variants (RD=1, RD=2):** We randomly sample 2 or 8 distractors from the original 18, with a fixed seed (42 + question hash) for reproducibility.

**The D6 interaction effect formula:**
```
drop_from_RC = accuracy(A) − accuracy(B)       # cost of hard reasoning alone
drop_from_RD = accuracy(A) − accuracy(C)       # cost of hard retrieval alone
predicted_additive = accuracy(A) − drop_RC − drop_RD   # if independent
actual = accuracy(E)                            # what actually happens
interaction_effect = predicted_additive − actual # positive = super-linear collapse
```

If `interaction_effect > 0.05`, the difficulties interact (compound). If ≈ 0, they're independent. If negative, they're redundant.

**Week 6 result:** Interaction effect = **+0.14 (14 percentage points)**. Super-linear collapse confirmed. RC alone costs 26pp, RD alone costs 10pp, but combined they cost 50pp (predicted additive: 36pp). The extra 14pp cannot be explained by either axis alone — the difficulties compound. 7 questions are "interaction victims" that pass B and C individually but fail E.

**Note on other difficulty axes:** Entity Ambiguity (EA) and Temporal Complexity (TC) are defined in the taxonomy but not yet tested — they require different dataset constructions. RC × RD is the most important interaction because it maps to the core agent loop: plan (RC) → retrieve (RD) → reason.

---

## Part 5 — Key variables and terms

### Trajectory

The complete record of what the agent did for one question. Like a play-by-play in sports. Includes:
- Every search query the agent sent
- Every document that came back
- How many tokens were used at each step
- Whether the step was a loop (same query as before)
- The final answer

```json
{
  "question": "What country is Bong Joon-ho from?",
  "trajectory": [
    {"step": 1, "query": "director of Parasite", "docs_returned": 3, "relevant": 1},
    {"step": 2, "query": "Bong Joon-ho nationality", "docs_returned": 3, "relevant": 1}
  ],
  "predicted_answer": "South Korea",
  "ground_truth": "South Korea",
  "correct": true
}
```

### Precision@k

How many of the k documents returned by a search were actually relevant. If k=3 and 1 document was relevant, precision@k = 1/3 = 0.333.

This is the raw input to D2.

### Hop

One step in a multi-step reasoning chain. A 2-hop question needs 2 pieces of information chained together. A 3-hop question needs 3.

Most questions in our benchmark are 2-hop. The agent consistently struggles even at 2 hops with Llama 3.1 8B.

### Supporting paragraphs

The specific paragraphs from the MuSiQue dataset that contain the information needed to answer the question. Used as (a) the knowledge base the agent searches, and (b) ground truth for judging whether a retrieval step was good.

### System prompt

The instructions we give the agent before it starts answering. In Week 2c and onwards, this includes explicit instructions like "you must search at least twice before giving an answer." The system prompt in Week 3 is what drove the step average from 1.38 to 2.02.

### Exact match

The strictest version of D1 — the agent's answer must match the ground truth string exactly (after normalisation). Our accuracy numbers (22%, 18%, etc.) use exact match. There's also a partial credit score for answers that contain the right words but not in the right form.

### Interference rate

Used in D4. Computed as `(correct_clean − correct_noisy) / correct_clean`. Positive values mean noise hurt the agent (expected). Negative values mean noise accidentally *helped* the agent (see: constructive interference). In Week 4, the interference rate was **−0.133** — the agent got more questions right on noisy data than clean.

### Query rewriting

A technique added in Week 4 to improve D2 (retrieval step quality). Before each FAISS search, the agent's raw query is passed to the LLM with a rewrite prompt: "Extract key entities and facts. Remove filler words." The rewritten query is used for the actual vector search, while the original query is preserved in the trajectory for D3 analysis.

**Effect:** D2 jumped from 0.183 (Week 3, no rewriting) to 0.302 (Week 4, with rewriting). Questions with D2=0 dropped from 26/50 to 15/48. Accuracy improved from 22% to 30%.

**Example:**
- Agent query: `"What is the name of the lead singer of the band Green Day?"`
- Rewritten: `"Green Day lead vocalist Billie Joe Armstrong"`

The rewritten query has higher entity density, which produces better vector similarity against the supporting paragraphs in FAISS.

### Cross-question swap

The noise injection method used for D4 in Week 4. For each question, 25% of its distractor paragraphs are replaced with supporting paragraphs from *other* questions. This creates plausible interference — real Wikipedia text about similar entity types, but containing facts that answer different questions. Supporting paragraphs for the current question are never replaced. Deterministic via `seed=42`.

### Constructive interference

An unexpected phenomenon observed in Week 4's D4 experiment. When supporting paragraphs from other questions were injected as "noise," they sometimes *improved* retrieval quality because they were higher-quality, entity-rich text compared to the random distractors they replaced. FAISS found them, and the LLM extracted enough context to reason to the right answer. This is the opposite of destructive interference (what we expected). It reveals that the agent's retrieval is so starved that any additional relevant-looking text helps.

### Degenerate output

A structurally broken answer that signals the agent failed before it even started reasoning:
- Raw JSON: `{"name": "search_knowledge_base", "parameters": ...}` — the LLM printed its tool call as text instead of executing it
- Refusal: `"Sorry, need more steps to process this request."`
- Empty: fewer than 5 characters

These are caught and forced to D1=0.0, D5=0.0 so they don't inflate scores.

### Undershooting

When the agent makes fewer searches than the question requires. A 2-hop question requires at least 2 searches — making only 1 is undershooting. This was the dominant failure mode in Week 2 (27/50 questions undershot). Fixed in Week 3 by enforcing minimum steps in the system prompt.

### Retrieval fixation

The Week 1 failure mode — the agent gets stuck searching the exact same query over and over. The opposite of undershooting (too few searches), retrieval fixation is too many identical searches. D3 catches this via its loop detection component.

---

## Part 6 — How the metrics relate to each other

The 5 active metrics form a diagnostic chain:

```
Is the question hard? (D6)
        ↓
Did the agent plan enough searches? (D3)
        ↓
Did each search find relevant documents? (D2)
        ↓
Did noise confuse the retrieval? (D4)
        ↓
Did the agent produce the right final answer? (D1)
                 + at what cost? (D5)
```

**D1 alone** tells you the outcome but not the cause.

**D1 + D2** lets you separate "failed because it retrieved nothing useful" from "failed because it had the right context but reasoned wrongly."

**D1 + D3** lets you separate "failed because it gave up too early" from "failed because its searches were bad."

**D1 + D2 + D3** gives you a full diagnostic. In Week 3:
- 9 questions hit the ideal path: good D3, good D2, correct D1
- 14 questions had good D3 (planned well) but D2=0 (retrieved nothing) → the embedding model is the bottleneck
- 2 questions were degenerate: D1=0, D2=0, D3=0, D5=0

**D1 + D2 + D3 + D4** (added in Week 4) reveals robustness. In Week 4:
- Query rewriting moved 9 questions from D2=0 to D2>0, and 4 of those became correct (D2→D1 pipeline confirmed)
- D4 showed noise *helped* the agent (interference rate = −0.133) — the agent is so retrieval-starved that any additional text improves it
- 2 of 4 "flipped" questions had identical D2 in clean and noisy — noise confused the LLM's reasoning, not retrieval. D4 catches a failure mode that D2 alone cannot see.

**D1 + D2 via hybrid retrieval** (Week 5) confirmed the causal link: adding BM25 to FAISS lifted D2 from 0.302→0.416 and accuracy from 30%→48%. D2=0 count dropped from 15→7. This is the strongest evidence that D2 *predicts* D1.

**RAGAS** sees only the final answer and final retrieved docs. It would see all 50 questions as high-scoring (fluent, on-topic answers) and have no way to distinguish the ideal-path questions from Type-C failures, nor detect constructive interference.

---

## Part 7 — What we're trying to prove

The benchmark is building toward one core claim for a research paper:

> **Claim:** Agentic RAG systems require multi-dimensional process evaluation, not just outcome evaluation. A system can score 0.927 on RAGAS while failing 100% of questions, getting stuck in 27-step loops, and spending 3,429 tokens per wrong answer. These failures are invisible to RAGAS but clearly captured by D1–D6.

Each week adds evidence:
- **Week 1:** The gap exists (RAGAS=0.927, accuracy=0%)
- **Week 2:** The gap is measurable (D1 + D5 as metric classes, ablation over k and prompting)
- **Week 3:** The gap is diagnostic (D2 + D3 pinpoint *where* failures happen — retrieval vs. planning)
- **Week 4:** The gap includes robustness — D4 reveals constructive interference (noise *helped* the agent, interference rate = −0.133), proving the agent is so retrieval-starved that even random helpful text improves it. Query rewriting shows D2→D1 pipeline: +65% retrieval → +36% accuracy. RAGAS can't measure any of this.
- **Week 5:** Hybrid retrieval confirms D2→D1 causation — BM25+FAISS lifts D2 from 0.302→0.416, accuracy from 30%→48%. D2=0 cut in half (15→7). 39/101 bad rewrites caught by validation.
- **Week 6:** D6 shows the gap includes difficulty interaction — RAGAS can't predict non-linear degradation when difficulty axes compound. 5 conditions × 50 questions. Interaction effect = +14pp (super-linear collapse confirmed). RC is 2.6× harder than RD. 7 "interaction victims" prove compounding.
- **Week 7–9:** Cross-system comparison will show the gap is consistent across different architectures

The final output is an arXiv preprint and a public leaderboard on HuggingFace Spaces.
