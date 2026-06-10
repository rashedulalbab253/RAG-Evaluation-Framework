"""
RAGAS Evaluator module.
Runs Faithfulness, Answer Relevancy, and Context Recall evaluations.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG outputs using RAGAS metrics with LLM fallback."""

    def __init__(self):
        api_key = Config.OPENAI_API_KEY
        if not api_key or "your-openai" in api_key.lower():
            api_key = "dummy-key-for-initialization"
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL, temperature=0, api_key=api_key
        )
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL, api_key=api_key
        )

    def prepare_dataset(self, rag_outputs, ground_truths):
        data = {
            "question": [r["question"] for r in rag_outputs],
            "answer": [r["answer"] for r in rag_outputs],
            "contexts": [r["contexts"] for r in rag_outputs],
            "ground_truth": ground_truths,
        }
        return Dataset.from_dict(data)

    def evaluate(self, rag_outputs, ground_truths):
        """Run RAGAS evaluation with fallback to LLM scoring."""
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_recall

            dataset = self.prepare_dataset(rag_outputs, ground_truths)
            logger.info(f"Running RAGAS evaluation on {len(rag_outputs)} samples...")
            results = ragas_evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
            )

            aggregate = {
                "faithfulness": float(results.get("faithfulness", 0)),
                "answer_relevancy": float(results.get("answer_relevancy", 0)),
                "context_recall": float(results.get("context_recall", 0)),
            }

            per_question = []
            results_df = results.to_pandas()
            for _, row in results_df.iterrows():
                per_question.append({
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "faithfulness": float(row.get("faithfulness", 0)),
                    "answer_relevancy": float(row.get("answer_relevancy", 0)),
                    "context_recall": float(row.get("context_recall", 0)),
                })

            return {"aggregate": aggregate, "per_question": per_question, "sample_count": len(rag_outputs)}

        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}. Using LLM fallback...")
            return self._fallback_evaluate(rag_outputs, ground_truths)

    def _fallback_evaluate(self, rag_outputs, ground_truths):
        per_question = []
        f_scores, r_scores, c_scores = [], [], []

        for i, (output, gt) in enumerate(zip(rag_outputs, ground_truths)):
            try:
                scores = self._llm_score(output, gt)
                per_question.append({"question": output["question"], "answer": output["answer"], "ground_truth": gt, **scores})
                f_scores.append(scores["faithfulness"])
                r_scores.append(scores["answer_relevancy"])
                c_scores.append(scores["context_recall"])
                if (i + 1) % 10 == 0:
                    logger.info(f"  Evaluated {i + 1}/{len(rag_outputs)}")
            except Exception as e:
                logger.warning(f"  Error scoring question {i + 1}: {e}")
                per_question.append({"question": output["question"], "answer": output["answer"], "ground_truth": gt, "faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0})
                f_scores.append(0.0); r_scores.append(0.0); c_scores.append(0.0)

        agg = {
            "faithfulness": sum(f_scores) / len(f_scores) if f_scores else 0,
            "answer_relevancy": sum(r_scores) / len(r_scores) if r_scores else 0,
            "context_recall": sum(c_scores) / len(c_scores) if c_scores else 0,
        }
        return {"aggregate": agg, "per_question": per_question, "sample_count": len(rag_outputs)}

    def _llm_score(self, output, ground_truth):
        ctx = "\n".join(output.get("contexts", []))[:2000]
        prompt = f"""Score this RAG output on three metrics (0.0-1.0 each):
1. Faithfulness: Is the answer grounded in the context?
2. Answer Relevancy: Does the answer address the question?
3. Context Recall: Does the context contain needed info?

Question: {output['question']}
Context: {ctx}
Answer: {output['answer']}
Ground Truth: {ground_truth}

Return ONLY JSON: {{"faithfulness": 0.X, "answer_relevancy": 0.X, "context_recall": 0.X}}"""

        response = self.llm.invoke(prompt)
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].split("```")[0]
            if content.startswith("json"):
                content = content[4:]
        scores = json.loads(content)
        return {k: min(1.0, max(0.0, float(scores.get(k, 0)))) for k in ["faithfulness", "answer_relevancy", "context_recall"]}

    def save_results(self, results, experiment_name="default", pipeline_config=None):
        Config.ensure_dirs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(Config.RESULTS_DIR, f"eval_{experiment_name}_{ts}.json")
        output = {"experiment_name": experiment_name, "timestamp": datetime.now().isoformat(), "pipeline_config": pipeline_config or {}, "results": results}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")
        return filepath
