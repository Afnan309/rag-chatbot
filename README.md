# RAG Chatbot Trained on Company Docs

<p align="center"><img src="docs/cover.jpg" width="820" alt="RAG Chatbot Trained on Company Docs"></p>

> Part of **[Afnan Asjad's AI portfolio](https://afnan309.github.io)** · [Hire me on Upwork](https://www.upwork.com/freelancers/~012b76801e57487390)

AI customer-support chatbot that answers only from a company knowledge base, cites every source, and refuses to guess.

## Features
- Section-aware chunking of Markdown documents
- Hybrid search: BM25 keyword index + vector embeddings, merged with Reciprocal Rank Fusion, then reranked
- Grounded answers with numbered citations
  - **OpenAI mode** (set `OPENAI_API_KEY`): `text-embedding-3-small` embeddings + GPT answer generation
  - **Offline mode** (no key): local LSA embeddings + extractive answers, free to run
- Guardrails: prompt-injection filter, grounding threshold (refuses when evidence is weak), staff-only documents hidden from customers
- Flask REST API + web chat UI with a live sources panel
- Evaluation script with a separate dev set (for tuning) and held-out test set (for reporting)

## Run
```bash
pip install -r requirements.txt
python app.py                 # http://localhost:8000
python eval/run_eval.py test  # held-out evaluation -> eval/results_test.json
```

Optional OpenAI mode:
```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
python app.py
```

## API
`POST /api/chat`  `{"message": "How much is an AC tune-up?", "audience": "customer"}`
returns `answer`, `status` (answered | fallback | blocked), `sources`, `latency_ms`.

## Results (held-out test set, offline mode)
| Metric | Value |
|---|---|
| Right passage in top 3 (hybrid) | 100% |
| Answer accuracy | 40/45 (88.9%) |
| Prompt injections blocked | 5/5 |
| Staff-only docs hidden from customers | 100% |
| Off-topic questions refused | 7/10 |
| Avg response time | ~3 ms (excludes LLM call) |

Numbers in OpenAI mode will differ; re-run `eval/run_eval.py` after setting the key.

## Project structure
```
rag/            chunker, bm25, embeddings, vector_store, hybrid, reranker, guardrails, generator, pipeline
data/docs/      sample knowledge base (9 documents, fictional home-services company)
eval/           dev_questions.json, test_questions.json, run_eval.py, results_*.json
templates/      chat UI
app.py          Flask API + UI
```

## Swapping in a hosted vector database
`rag/vector_store.py` exposes `add()` and `search()`; replace it with a Pinecone, Qdrant or pgvector client to scale beyond memory.

## Screenshots

| | |
|---|---|
| ![Screen 1](docs/screen_1.jpg) | ![Screen 2](docs/screen_2.jpg) |
| ![Screen 3](docs/screen_3.jpg) | ![Results](docs/results.jpg) |
