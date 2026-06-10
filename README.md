# RAG Evaluation Framework

An automated evaluation suite that scores RAG (Retrieval-Augmented Generation) pipelines across three critical dimensions using **RAGAS** metrics. Developed by **Rashedul Albab**.

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (hallucination detection) |
| **Answer Relevancy** | Does the answer actually address the question asked? |
| **Context Recall** | Did the retriever surface the right documents? |

## Tech Stack

- **RAG Pipeline**: LangChain + ChromaDB + OpenAI
- **Evaluation**: RAGAS (Faithfulness, Answer Relevancy, Context Recall)
- **Test Data**: Synthetic QA generation from Wikipedia articles
- **Dashboard**: Flask + Vanilla JS with real-time pipeline monitoring
- **Trade-off Analysis**: Compare configurations side-by-side

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Launch the dashboard
python app.py
# Open http://localhost:5050
```

## Project Structure

```
├── app.py                     # Flask dashboard server
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
├── src/
│   ├── data_loader.py         # Wikipedia document loader + chunking
│   ├── rag_pipeline.py        # LangChain + ChromaDB RAG pipeline
│   ├── testset_generator.py   # Synthetic QA test set generation
│   ├── evaluator.py           # RAGAS evaluation engine
│   └── experiment_tracker.py  # Experiment comparison & trade-offs
├── templates/
│   └── index.html             # Dashboard UI
├── static/
│   ├── css/style.css          # Dashboard styles
│   └── js/dashboard.js        # Dashboard logic
├── data/                      # Generated datasets (auto-created)
└── results/                   # Evaluation results (auto-created)
```

## How It Works

1. **Load Documents** — Fetches Wikipedia articles on AI/ML topics
2. **Chunk Documents** — Splits into configurable-size chunks with overlap
3. **Generate Test Set** — Creates synthetic QA pairs using RAGAS or LLM fallback
4. **Build Index** — Embeds chunks into ChromaDB vector store
5. **Run RAG** — Queries the pipeline on every test question
6. **Evaluate** — Scores each answer on all three RAGAS metrics
7. **Compare** — Run again with different settings to see trade-offs

## Trade-off Analysis

The key insight: changing one parameter often improves one metric while hurting another.

| Change | Faithfulness | Relevancy | Recall |
|---|---|---|---|
| Smaller chunks | ↑ More precise | → Neutral | ↓ Less context |
| Larger chunks | ↓ More noise | → Neutral | ↑ More coverage |
| Higher k | ↓ More noise | → Neutral | ↑ More docs found |
| Lower k | ↑ Less noise | → Neutral | ↓ Fewer docs |

## Resume Bullet Points (Copy-Paste Ready)

- **Designed and implemented an automated RAG evaluation suite** using the **RAGAS** framework, scoring pipeline performance across three key dimensions: Faithfulness (hallucination detection), Answer Relevancy, and Context Recall.
- **Developed a synthetic QA dataset generator** that automatically constructs question-answer test sets from raw document corpora (Wikipedia) utilizing RAGAS Knowledge Graph evolutions, eliminating the need for manually labeled evaluation data.
- **Built an interactive dashboard and experiment tracker** (Flask, vanilla JS) to evaluate parameter sweeps (chunk size, retrieval depth $k$, prompts) and analyze performance trade-offs (e.g., verifying that smaller chunks optimized Faithfulness to 0.86 but decreased Context Recall to 0.66).

## License

MIT - Copyright (c) 2026 Rashedul Albab
