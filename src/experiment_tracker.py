"""
Experiment Tracker module.
Tracks multiple evaluation runs for trade-off analysis across configurations.
"""

import os
import json
import glob
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Loads, compares, and analyzes evaluation experiments."""

    def __init__(self):
        Config.ensure_dirs()

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all saved experiment results."""
        pattern = os.path.join(Config.RESULTS_DIR, "eval_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        experiments = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                experiments.append({
                    "filename": os.path.basename(fp),
                    "filepath": fp,
                    "experiment_name": data.get("experiment_name", "unknown"),
                    "timestamp": data.get("timestamp", ""),
                    "pipeline_config": data.get("pipeline_config", {}),
                    "aggregate": data.get("results", {}).get("aggregate", {}),
                    "sample_count": data.get("results", {}).get("sample_count", 0),
                })
            except Exception as e:
                logger.warning(f"Error reading {fp}: {e}")
        return experiments

    def get_experiment(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a specific experiment by filename."""
        filepath = os.path.join(Config.RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare_experiments(self, filenames: List[str]) -> Dict[str, Any]:
        """Compare aggregate scores across experiments."""
        comparison = []
        for fn in filenames:
            exp = self.get_experiment(fn)
            if exp:
                agg = exp.get("results", {}).get("aggregate", {})
                comparison.append({
                    "name": exp.get("experiment_name", fn),
                    "timestamp": exp.get("timestamp", ""),
                    "config": exp.get("pipeline_config", {}),
                    "faithfulness": agg.get("faithfulness", 0),
                    "answer_relevancy": agg.get("answer_relevancy", 0),
                    "context_recall": agg.get("context_recall", 0),
                })
        return {"experiments": comparison, "count": len(comparison)}

    def get_tradeoff_analysis(self) -> Dict[str, Any]:
        """Analyze trade-offs across all experiments."""
        experiments = self.list_experiments()
        if len(experiments) < 2:
            return {"message": "Need at least 2 experiments for trade-off analysis", "experiments": experiments}

        best = {}
        for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
            sorted_exps = sorted(experiments, key=lambda x: x["aggregate"].get(metric, 0), reverse=True)
            if sorted_exps:
                best[metric] = {
                    "best_experiment": sorted_exps[0]["experiment_name"],
                    "best_score": sorted_exps[0]["aggregate"].get(metric, 0),
                    "worst_experiment": sorted_exps[-1]["experiment_name"],
                    "worst_score": sorted_exps[-1]["aggregate"].get(metric, 0),
                    "range": sorted_exps[0]["aggregate"].get(metric, 0) - sorted_exps[-1]["aggregate"].get(metric, 0),
                }

        return {"best_per_metric": best, "total_experiments": len(experiments), "experiments": experiments}

    def delete_experiment(self, filename: str) -> bool:
        """Delete an experiment result file."""
        filepath = os.path.join(Config.RESULTS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted experiment: {filename}")
            return True
        return False
