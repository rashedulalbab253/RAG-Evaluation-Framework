"""
Flask Web Dashboard for the RAG Evaluation Framework.
Provides API endpoints and serves the evaluation dashboard UI.
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from src.data_loader import DocumentLoader
from src.rag_pipeline import RAGPipeline
from src.testset_generator import TestSetGenerator
from src.evaluator import RAGEvaluator
from src.experiment_tracker import ExperimentTracker

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask App ───────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
Config.ensure_dirs()

tracker = ExperimentTracker()

# Global state for async pipeline runs
pipeline_status = {
    "running": False,
    "stage": "idle",
    "progress": 0,
    "message": "",
    "error": None,
}


def update_status(stage, progress, message, error=None):
    global pipeline_status
    pipeline_status = {
        "running": stage != "complete" and stage != "error",
        "stage": stage,
        "progress": progress,
        "message": message,
        "error": error,
    }
    logger.info(f"[{stage}] {message} ({progress}%)")


# ── Page Routes ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API Routes ──────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)


@app.route("/api/experiments")
def api_experiments():
    experiments = tracker.list_experiments()
    return jsonify({"experiments": experiments})


@app.route("/api/experiment/<filename>")
def api_experiment_detail(filename):
    exp = tracker.get_experiment(filename)
    if exp is None:
        return jsonify({"error": "Experiment not found"}), 404
    return jsonify(exp)


@app.route("/api/tradeoff")
def api_tradeoff():
    analysis = tracker.get_tradeoff_analysis()
    return jsonify(analysis)


@app.route("/api/experiment/<filename>", methods=["DELETE"])
def api_delete_experiment(filename):
    success = tracker.delete_experiment(filename)
    return jsonify({"success": success})


@app.route("/api/run", methods=["POST"])
def api_run_pipeline():
    """Run the full pipeline: load docs → generate testset → RAG → evaluate."""
    global pipeline_status
    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    data = request.json or {}
    experiment_name = data.get("experiment_name", "experiment")
    chunk_size = int(data.get("chunk_size", Config.CHUNK_SIZE))
    chunk_overlap = int(data.get("chunk_overlap", Config.CHUNK_OVERLAP))
    retrieval_k = int(data.get("retrieval_k", Config.RETRIEVAL_K))
    testset_size = int(data.get("testset_size", min(Config.TESTSET_SIZE, 30)))
    topics = data.get("topics", Config.DEFAULT_TOPICS[:3])
    prompt_template = data.get("prompt_template", None)

    def run_pipeline():
        try:
            # Stage 1: Load documents
            update_status("loading", 5, "Loading Wikipedia articles...")
            loader = DocumentLoader(
                topics=topics, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            raw_docs = loader.load_wikipedia_articles()
            update_status("loading", 15, f"Loaded {len(raw_docs)} articles")

            # Stage 2: Chunk documents
            update_status("chunking", 20, f"Chunking with size={chunk_size}, overlap={chunk_overlap}...")
            chunks = loader.chunk_documents()
            loader.save_documents_metadata()
            update_status("chunking", 30, f"Created {len(chunks)} chunks")

            # Stage 3: Generate test set
            update_status("generating", 35, f"Generating {testset_size} synthetic QA pairs...")
            gen = TestSetGenerator(testset_size=testset_size)
            testset = gen.generate_and_save(raw_docs)
            update_status("generating", 50, f"Generated {len(testset)} QA pairs")

            # Stage 4: Build RAG pipeline
            update_status("indexing", 55, "Building vector index with ChromaDB...")
            pipeline_kwargs = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_k": retrieval_k,
                "collection_name": f"eval_{experiment_name}",
            }
            if prompt_template:
                pipeline_kwargs["prompt_template"] = prompt_template

            pipeline = RAGPipeline(**pipeline_kwargs)
            pipeline.ingest_documents(chunks)
            update_status("indexing", 65, "Vector index built")

            # Stage 5: Run RAG on test set
            update_status("querying", 70, "Running RAG pipeline on test questions...")
            questions = [item["question"] for item in testset]
            rag_outputs = pipeline.batch_query(questions)
            update_status("querying", 80, f"Completed {len(rag_outputs)} queries")

            # Stage 6: Evaluate
            update_status("evaluating", 85, "Running RAGAS evaluation...")
            evaluator = RAGEvaluator()
            ground_truths = [item.get("ground_truth", "") for item in testset]
            results = evaluator.evaluate(rag_outputs, ground_truths)
            update_status("evaluating", 95, "Evaluation complete")

            # Stage 7: Save results
            config_used = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_k": retrieval_k,
                "topics": topics,
                "testset_size": testset_size,
                "model": Config.OPENAI_MODEL,
            }
            evaluator.save_results(results, experiment_name, config_used)

            # Cleanup
            pipeline.cleanup()

            agg = results.get("aggregate", {})
            msg = (
                f"Done! F={agg.get('faithfulness', 0):.2f} "
                f"AR={agg.get('answer_relevancy', 0):.2f} "
                f"CR={agg.get('context_recall', 0):.2f}"
            )
            update_status("complete", 100, msg)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            update_status("error", 0, str(e), error=str(e))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    return jsonify({"message": "Pipeline started", "experiment_name": experiment_name})


@app.route("/api/demo-results", methods=["POST"])
def api_load_demo():
    """Load demonstration results for dashboard preview."""
    import random
    random.seed(42)

    configs = [
        {"name": "baseline", "chunk_size": 1000, "chunk_overlap": 200, "retrieval_k": 4},
        {"name": "small_chunks", "chunk_size": 500, "chunk_overlap": 100, "retrieval_k": 4},
        {"name": "large_chunks", "chunk_size": 2000, "chunk_overlap": 300, "retrieval_k": 4},
        {"name": "deep_retrieval", "chunk_size": 1000, "chunk_overlap": 200, "retrieval_k": 8},
        {"name": "shallow_retrieval", "chunk_size": 1000, "chunk_overlap": 200, "retrieval_k": 2},
    ]

    sample_questions = [
        "What is artificial intelligence?",
        "How does machine learning differ from traditional programming?",
        "What are neural networks?",
        "Explain the transformer architecture.",
        "What is natural language processing?",
        "How does deep learning work?",
        "What are large language models?",
        "What is retrieval-augmented generation?",
        "How do attention mechanisms work?",
        "What is transfer learning?",
        "Explain backpropagation.",
        "What is a convolutional neural network?",
        "How does BERT work?",
        "What is GPT?",
        "Explain the concept of embeddings.",
    ]

    score_profiles = {
        "baseline":          {"f": (0.78, 0.12), "ar": (0.82, 0.10), "cr": (0.75, 0.15)},
        "small_chunks":      {"f": (0.85, 0.08), "ar": (0.80, 0.11), "cr": (0.65, 0.18)},
        "large_chunks":      {"f": (0.70, 0.15), "ar": (0.84, 0.09), "cr": (0.82, 0.12)},
        "deep_retrieval":    {"f": (0.72, 0.14), "ar": (0.79, 0.12), "cr": (0.88, 0.08)},
        "shallow_retrieval": {"f": (0.82, 0.10), "ar": (0.85, 0.08), "cr": (0.58, 0.20)},
    }

    for cfg in configs:
        profile = score_profiles[cfg["name"]]
        per_q = []
        for q in sample_questions:
            per_q.append({
                "question": q,
                "answer": f"[Demo answer for '{q[:30]}...']",
                "ground_truth": f"[Demo ground truth for '{q[:30]}...']",
                "faithfulness": max(0, min(1, random.gauss(profile["f"][0], profile["f"][1]))),
                "answer_relevancy": max(0, min(1, random.gauss(profile["ar"][0], profile["ar"][1]))),
                "context_recall": max(0, min(1, random.gauss(profile["cr"][0], profile["cr"][1]))),
            })

        agg = {
            "faithfulness": sum(p["faithfulness"] for p in per_q) / len(per_q),
            "answer_relevancy": sum(p["answer_relevancy"] for p in per_q) / len(per_q),
            "context_recall": sum(p["context_recall"] for p in per_q) / len(per_q),
        }

        results = {"aggregate": agg, "per_question": per_q, "sample_count": len(per_q)}
        evaluator = RAGEvaluator()
        evaluator.save_results(results, cfg["name"], cfg)

    return jsonify({"message": "Demo results loaded", "count": len(configs)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)
