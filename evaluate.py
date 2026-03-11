"""
Evaluation Script for Neuro-Symbolic UXO Framework.

This script evaluates the pipeline on a dataset and computes metrics:
- Precision, Recall (Macro and Normal)
- F1-Score (Global, counts U as misclassification)
- F1-Macro (mean of per-class F1)
- F1-Confident (computed on non-U samples)
- Human-in-the-Loop Request Rate (HRR)

Evaluation is run N times (default 100) to compute mean ± std.

Reference: Paper Section 5 - Experiments, Table 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.types import SystemState


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    precision_normal: float = 0.0
    recall_normal: float = 0.0
    f1_score: float = 0.0
    f1_macro: float = 0.0
    f1_confident: float = 0.0
    hrr: float = 0.0
    accuracy: float = 0.0
    fnr: float = 0.0


@dataclass
class EvaluationResult:
    """Result of a single evaluation run."""
    metrics: EvaluationMetrics
    predictions: List[Dict] = field(default_factory=list)
    confusion_matrix: Optional[np.ndarray] = None


def load_dataset(path: str) -> List[Dict]:
    """
    Load evaluation dataset.
    
    Expected format:
    [
        {"image_path": "...", "class_name": "Mortar_Bomb"},
        ...
    ]
    
    Args:
        path: Path to JSON dataset file
    
    Returns:
        List of sample dictionaries
    """
    with open(path, 'r') as f:
        return json.load(f)


def make_derived_path(base_path: str, suffix: str) -> str:
    """
    Create a derived JSON path based on a base path.
    If base_path ends with .json, insert suffix before the extension.
    Otherwise, append suffix and .json.
    """
    if base_path.endswith(".json"):
        return f"{base_path[:-5]}{suffix}.json"
    return f"{base_path}{suffix}.json"


def load_existing_results(output_path: str) -> Tuple[List[Dict], set]:
    """
    Load existing results from output file if it exists.
    
    Args:
        output_path: Path to the output JSON file
    
    Returns:
        Tuple of (existing_samples_list, set_of_processed_image_paths)
    """
    if not os.path.exists(output_path):
        return [], set()
    
    try:
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
        
        samples = existing_data.get('samples', [])
        processed_paths = {s.get('image_path') for s in samples if s.get('image_path')}
        
        print(f"[RESUME] Found existing results: {len(samples)} samples processed")
        return samples, processed_paths
    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[RESUME] Warning: Could not parse existing file: {e}")
        print(f"[RESUME] Starting fresh...")
        return [], set()


def compute_metrics(
    predictions: List[str],
    ground_truth: List[str],
    states: List[SystemState],
    risks: Optional[List[str]] = None
) -> EvaluationMetrics:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of predicted class names
        ground_truth: List of ground truth class names
        states: List of system states (NORMAL/UNCERTAINTY)
    
    Returns:
        EvaluationMetrics instance
    """
    
    total_samples = len(predictions)
    if total_samples == 0:
        return EvaluationMetrics()

    valid_pairs = [
        (p, g) for p, g in zip(predictions, ground_truth)
        if g is not None
    ]
    total_with_gt = len(valid_pairs)

    correct_global = sum(1 for p, g in valid_pairs if p == g)
    accuracy = correct_global / total_with_gt if total_with_gt > 0 else 0.0

    classes = sorted({g for _, g in valid_pairs})
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    for cls in classes:
        tp = sum(1 for p, g in valid_pairs if p == cls and g == cls)
        fp = sum(1 for p, g in valid_pairs if p == cls and g != cls)
        fn = sum(1 for p, g in valid_pairs if g == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precision_per_class.append(prec)
        recall_per_class.append(rec)
        f1_per_class.append(f1)

    macro_precision = float(np.mean(precision_per_class)) if precision_per_class else 0.0
    macro_recall = float(np.mean(recall_per_class)) if recall_per_class else 0.0
    f1_macro = float(np.mean(f1_per_class)) if f1_per_class else 0.0

    # Compute weighted F1-Score (harmonic mean weighted by class support)
    total_support = sum(
        sum(1 for _, g in valid_pairs if g == cls) for cls in classes
    )
    f1_weighted = sum(
        f1_c * sum(1 for _, g in valid_pairs if g == cls)
        for f1_c, cls in zip(f1_per_class, classes)
    ) / total_support if total_support > 0 else 0.0
    f1_score = f1_weighted

    # Compute global recall (weighted by class support)
    recall_weighted = sum(
        rec_c * sum(1 for _, g in valid_pairs if g == cls)
        for rec_c, cls in zip(recall_per_class, classes)
    ) / total_support if total_support > 0 else 0.0

    confident_pairs = [
        (p, g) for p, g, s in zip(predictions, ground_truth, states)
        if s == SystemState.NORMAL and g is not None
    ]
    if confident_pairs:
        conf_classes = sorted({g for _, g in confident_pairs})
        conf_prec_per_class = []
        conf_rec_per_class = []
        conf_f1_per_class = []
        for cls in conf_classes:
            tp = sum(1 for p, g in confident_pairs if p == cls and g == cls)
            fp = sum(1 for p, g in confident_pairs if p == cls and g != cls)
            fn = sum(1 for p, g in confident_pairs if g == cls and p != cls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            conf_prec_per_class.append(prec)
            conf_rec_per_class.append(rec)
            conf_f1_per_class.append(f1)
        precision_normal = float(np.mean(conf_prec_per_class)) if conf_prec_per_class else 0.0
        recall_normal = float(np.mean(conf_rec_per_class)) if conf_rec_per_class else 0.0
        f1_confident = float(np.mean(conf_f1_per_class)) if conf_f1_per_class else 0.0
    else:
        precision_normal = 0.0
        recall_normal = 0.0
        f1_confident = 0.0

    uncertain_count = sum(1 for s in states if s == SystemState.UNCERTAINTY)
    hrr = uncertain_count / total_samples if total_samples > 0 else 0.0

    fnr = 0.0
    if risks:
        high_risk_labels = {"high", "critical"}
        fn = 0
        tp = 0
        for p, g, s, r in zip(predictions, ground_truth, states, risks):
            if r is None:
                continue
            if str(r).lower() in high_risk_labels:
                if s != SystemState.NORMAL:
                    continue
                if p == g:
                    tp += 1
                else:
                    fn += 1
        denom = fn + tp
        fnr = fn / denom if denom > 0 else 0.0

    return EvaluationMetrics(
        precision=macro_precision,
        recall=recall_weighted,
        precision_normal=precision_normal,
        recall_normal=recall_normal,
        f1_score=f1_score,
        f1_macro=f1_macro,
        f1_confident=f1_confident,
        hrr=hrr,
        accuracy=accuracy,
        fnr=fnr
    )


def run_single_evaluation(
    pipeline: NeuroSymbolicPipeline,
    dataset: List[Dict],
    verbose: bool = False,
    output_path: Optional[str] = None,
    resume: bool = True
) -> EvaluationResult:
    """
    Run a single evaluation pass.
    
    Args:
        pipeline: The neuro-symbolic pipeline
        dataset: Evaluation dataset
        verbose: Whether to print progress
    
    Returns:
        EvaluationResult
    """
    existing_results = []
    processed_paths = set()
    
    if resume and output_path:
        existing_results, processed_paths = load_existing_results(output_path)
        if processed_paths:
            print(f"[RESUME] Will skip {len(processed_paths)} already processed samples")
    
    predictions = []
    ground_truth = []
    states = []
    risks = []
    all_results = list(existing_results)
    timing_stats = {'total': [], 'vlm_inference': [], 'kg_reasoning': [], 'psl_inference': [], 'feedback_loop': []}
    
    risk_by_path = {s["image_path"]: s.get("risk") for s in dataset if s.get("image_path")}

    for existing in existing_results:
        if 'predicted' in existing and 'ground_truth' in existing:
            predictions.append(existing['predicted'].get('class', 'UNKNOWN'))
            gt_info = existing.get('ground_truth', {})
            ground_truth.append(gt_info.get('class', 'UNKNOWN'))
            state_val = existing.get('state', 'normal')
            states.append(SystemState.UNCERTAINTY if state_val == 'uncertainty' else SystemState.NORMAL)
            risk_val = gt_info.get('risk')
            if risk_val is None:
                risk_val = risk_by_path.get(existing.get('image_path'))
            risks.append(risk_val)
    
    skipped_count = 0
    
    for i, sample in enumerate(dataset):
        image_path = sample["image_path"]
        gt_class = sample["class_name"]
        gt_risk = sample.get("risk", "High")
        
        if image_path in processed_paths:
            skipped_count += 1
            if skipped_count <= 5 or skipped_count % 100 == 0:
                print(f"  [SKIP] {image_path} (already processed)")
            continue
        
        if verbose and (i % 100 == 0 or i == 0 or i == 1):
            print(f"  Processing {i}/{len(dataset)} (new: {len(all_results) - len(existing_results) + 1})...")
        
        try:
            print(image_path)
            result = pipeline.run(image_path)
            
            predictions.append(result.predicted_class)
            ground_truth.append(gt_class)
            states.append(result.state)
            risks.append(gt_risk)
            print("Prediction: ",result.predicted_class,"\n")
            print("GT: ",gt_class,"\n")
            print("State: ",result.state,"\n")
            print("Confidence: ",result.confidence,"\n")
            print("Iterations: ",result.iterations_used)
            reasoning_trace = result.reasoning_trace
            print(reasoning_trace)
            all_results.append({
                "sample_index": i + 1,
                "image_path": image_path,
                "ground_truth": {"class": gt_class, "risk": gt_risk},
                "predicted": {"class": result.predicted_class},
                "correct": result.predicted_class == gt_class,
                "state": result.state.value,
                "confidence": result.confidence,
                "iterations": result.iterations_used,
                "vlm_attributes": result.attribute_confidences,
                "kg_energies": result.kg_energies,
                "psl_scores": result.psl_scores,
                "feedback_details": result.feedback_details,
                "reasoning_trace": result.reasoning_trace,
                "timing_ms": result.timing_ms,
                "vlm_trace": getattr(result, "vlm_trace", [])
            })
            
            for key in timing_stats:
                if key in result.timing_ms:
                    timing_stats[key].append(result.timing_ms[key])
            
        except TimeoutError as e:
            print(f"  TIMEOUT: {str(e)}")
            predictions.append("TIMEOUT")
            ground_truth.append(gt_class)
            states.append(SystemState.UNCERTAINTY)
            risks.append(gt_risk)
            
            all_results.append({
                "sample_index": i + 1,
                "image_path": image_path,
                "ground_truth": {"class": gt_class, "risk": gt_risk},
                "predicted": {"class": "TIMEOUT"},
                "correct": False,
                "error": "timeout",
                "error_message": str(e)
            })
            
        except Exception as e:
            predictions.append("ERROR")
            ground_truth.append(gt_class)
            states.append(SystemState.UNCERTAINTY)
            risks.append(gt_risk)
            
            all_results.append({
                "image_path": image_path,
                "ground_truth": {"class": gt_class, "risk": gt_risk},
                "error": str(e)
            })
            
        # Save after EACH sample to output file
        if output_path:
            try:
                current_metrics = compute_metrics(
                    predictions, ground_truth,
                    states, risks
                )
                
                output_data = {
                    "metadata": {
                        "samples_processed": len(all_results),
                        "total_samples": len(dataset)
                    },
                    "live_metrics": {
                        "accuracy": current_metrics.accuracy,
                    "precision_macro": current_metrics.precision,
                    "recall_macro": current_metrics.recall,
                    "precision_normal": current_metrics.precision_normal,
                    "recall_normal": current_metrics.recall_normal,
                    "f1_score": current_metrics.f1_score,
                    "f1_macro": current_metrics.f1_macro,
                    "f1_confident": current_metrics.f1_confident,
                    "hrr": current_metrics.hrr,
                    "fnr_high_risk": current_metrics.fnr
                },
                    "samples": all_results
                }
                
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                if (i + 1) % 50 == 0:
                    snapshot_path = make_derived_path(output_path, f"_snapshot_{i+1}")
                    try:
                        with open(snapshot_path, 'w') as snap_f:
                            json.dump(output_data, snap_f, indent=2, default=str)
                        print(f"  Saved snapshot at {i+1} samples to {snapshot_path}")
                    except Exception as snap_error:
                        print(f"  Warning: Failed to save snapshot: {snap_error}")
                    
            except Exception as save_error:
                print(f"  Warning: Failed to save results: {save_error}")
    
    metrics = compute_metrics(
        predictions, ground_truth,
        states, risks
    )
    
    if output_path:
        secondary_path = make_derived_path(output_path, "_secondary_metrics")
        feedback_triggered = sum(1 for r in all_results if r.get('feedback_details', {}).get('triggered', False))
        feedback_iterations = [r.get('feedback_details', {}).get('iterations', 0) for r in all_results if r.get('feedback_details', {}).get('triggered', False)]

        num_classes = len(pipeline.kg.classes)
        num_attributes = len(getattr(pipeline.kg, "active_attributes", pipeline.kg.all_attributes))
        nodes = num_classes + num_attributes
        edges = pipeline.kg.count_edges()
        grounded = num_classes * num_attributes
        
        secondary_metrics = {
            "timing_statistics": {
                key: {
                    "mean": float(np.mean(values)) if values else 0.0,
                    "std": float(np.std(values)) if values else 0.0,
                    "min": float(min(values)) if values else 0.0,
                    "max": float(max(values)) if values else 0.0
                }
                for key, values in timing_stats.items() if values
            },
            "kg_complexity": {
                "theoretical": f"O(|Y| x |A|) = O({num_classes} x {num_attributes}) = O({grounded})",
                "nodes": nodes,
                "edges": edges,
                "grounded_facts_per_image": grounded,
                "measured_avg_ms": float(np.mean(timing_stats['kg_reasoning'])) if timing_stats['kg_reasoning'] else 0.0
            },
            "feedback_statistics": {
                "trigger_rate": feedback_triggered / len(all_results) if all_results else 0.0,
                "avg_iterations": float(np.mean(feedback_iterations)) if feedback_iterations else 0.0,
                "total_triggered": feedback_triggered
            }
        }
        
        try:
            with open(secondary_path, 'w') as f:
                json.dump(secondary_metrics, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save secondary metrics: {e}")
    
    return EvaluationResult(
        metrics=metrics,
        predictions=all_results
    )


def run_evaluation(
    dataset_path: str,
    provider: str = "mock",
    num_runs: int = 100,
    kg_path: str = "data/knowledge_graph.json",
    output_path: Optional[str] = None,
    verbose: bool = True,
    **vlm_kwargs
) -> Dict:
    """
    Run full evaluation with multiple runs.
    
    Args:
        dataset_path: Path to evaluation dataset
        provider: VLM provider type
        num_runs: Number of evaluation runs
        kg_path: Path to Knowledge Graph
        output_path: Optional path to save results
        verbose: Whether to print progress
        **vlm_kwargs: Additional VLM provider arguments
    
    Returns:
        Dictionary with mean and std for all metrics
    """
    if verbose:

        print("  Neuro-Symbolic UXO Framework - Evaluation")

        print(f"Dataset: {dataset_path}")
        print(f"Provider: {provider}")
        print(f"Number of runs: {num_runs}")

    
    dataset = load_dataset(dataset_path)
    if verbose:
        print(f"Loaded {len(dataset)} samples")
    
    from pipeline import NeuroSymbolicPipeline
    pipeline = NeuroSymbolicPipeline(
        kg_path=kg_path,
        vlm_provider=provider,
        **vlm_kwargs
    )
    
    all_metrics = {
        "precision": [],
        "recall": [],
        "precision_normal": [],
        "recall_normal": [],
        "f1_score": [],
        "f1_macro": [],
        "f1_confident": [],
        "hrr": [],
        "accuracy": [],
        "fnr": []
    }
    
    for run_idx in range(num_runs):
        if verbose:
            print(f"\nRun {run_idx + 1}/{num_runs}")
        
        result = run_single_evaluation(
            pipeline, 
            dataset, 
            verbose=True,
            output_path=output_path if num_runs == 1 else None # Only save intermediate for single runs to avoid spam
        )
        
        all_metrics["precision"].append(result.metrics.precision)
        all_metrics["recall"].append(result.metrics.recall)
        all_metrics["precision_normal"].append(result.metrics.precision_normal)
        all_metrics["recall_normal"].append(result.metrics.recall_normal)
        all_metrics["f1_score"].append(result.metrics.f1_score)
        all_metrics["f1_macro"].append(result.metrics.f1_macro)
        all_metrics["f1_confident"].append(result.metrics.f1_confident)
        all_metrics["hrr"].append(result.metrics.hrr)
        all_metrics["accuracy"].append(result.metrics.accuracy)
        all_metrics["fnr"].append(result.metrics.fnr)
    
    
    final_results = {}
    for metric_name, values in all_metrics.items():
        final_results[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    
    if verbose:
        print("\n" + "=" * 30)
        print("EVALUATION RESULTS")

        print(f"{'Metric':<20} {'Mean':>10} {'± Std':>12}")
        print(f"{'Precision (Macro)':<20} {final_results['precision']['mean']:>10.4f} ± {final_results['precision']['std']:.4f}")
        print(f"{'Precision (Normal)':<20} {final_results['precision_normal']['mean']:>10.4f} ± {final_results['precision_normal']['std']:.4f}")
        print(f"{'Recall (Macro)':<20} {final_results['recall']['mean']:>10.4f} ± {final_results['recall']['std']:.4f}")
        print(f"{'Recall (Normal)':<20} {final_results['recall_normal']['mean']:>10.4f} ± {final_results['recall_normal']['std']:.4f}")
        print(f"{'F1-Score':<20} {final_results['f1_score']['mean']:>10.4f} ± {final_results['f1_score']['std']:.4f}")
        print(f"{'F1-Macro':<20} {final_results['f1_macro']['mean']:>10.4f} ± {final_results['f1_macro']['std']:.4f}")
        print(f"{'F1-Confident':<20} {final_results['f1_confident']['mean']:>10.4f} ± {final_results['f1_confident']['std']:.4f}")
        print(f"{'HRR':<20} {final_results['hrr']['mean']*100:>9.2f}% ± {final_results['hrr']['std']*100:.2f}%")
        print(f"{'FNR (High)':<20} {final_results['fnr']['mean']*100:>9.2f}% ± {final_results['fnr']['std']*100:.2f}%")
    
    
    if output_path:
        metrics_path = make_derived_path(output_path, "_metrics")
        with open(metrics_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {metrics_path}")
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Neuro-Symbolic UXO Framework"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="data/test_set.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="mock",
        choices=["mock", "openai", "local", "transformers", "nscale", "huggingface"],
        help="VLM provider: mock (testing), openai (API), local (vLLM), nscale (Llama 4), huggingface (HF Inference API)"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=1,
        help="Number of evaluation runs"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path or HuggingFace model ID (required for transformers provider)"
    )
    parser.add_argument(
        "--model", "--hf-model",
        dest="hf_model",
        type=str,
        default=None,
        help="HuggingFace Inference API model name (for provider huggingface)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline evaluation (direct VLM, no KG/PSL)"
    )
    parser.add_argument(
        "--baseline-attributes",
        action="store_true",
        help="Run baseline evaluation WITH attributes constraints (Zero Shot + Attributes)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and neuro-symbolic for comparison"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset for testing"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Size of sample dataset to create"
    )
    
    args = parser.parse_args()
    vlm_kwargs = {}
    if args.model_path:
        vlm_kwargs["model_path"] = args.model_path
    if args.provider == "huggingface" and args.hf_model:
        vlm_kwargs["model"] = args.hf_model

    if args.compare:
        print("  COMPARATIVE EVALUATION: Baseline vs Neuro-Symbolic")
        from pipeline import NeuroSymbolicPipeline
        
        dataset = load_dataset(args.dataset)
        output_base = args.output.replace('.json', '') if args.output else 'results/comparison'
        baseline_output = f"{output_base}_baseline.json"
        neuro_output = f"{output_base}_neurosymbolic.json"
        
        from baseline import BaselinePipeline
        baseline_pipeline = BaselinePipeline(
            vlm_provider=args.provider,
            **vlm_kwargs
        )
        neuro_pipeline = NeuroSymbolicPipeline(
            vlm_provider=args.provider,
            **vlm_kwargs
        )
        
        baseline_results = []
        neuro_results = []
        baseline_preds, baseline_gt, baseline_states = [], [], []
        neuro_preds, neuro_gt, neuro_states = [], [], []
        baseline_risks, neuro_risks = [], []
        timing_stats = {'total': [], 'vlm_inference': [], 'kg_reasoning': [], 'psl_inference': [], 'feedback_loop': []}
        
        for i, sample in enumerate(dataset):
            print(f"\n[{i+1}/{len(dataset)}] Processing: {sample['image_path']}")
            gt_class = sample["class_name"]
            gt_risk = sample.get("risk")
            
            print("BASELINE...")
            baseline_result = baseline_pipeline.run(sample["image_path"])
            baseline_preds.append(baseline_result.predicted_class)
            baseline_gt.append(gt_class)
            baseline_states.append(baseline_result.state)
            baseline_risks.append(gt_risk)
            
            prompt_given = baseline_result.vlm_trace[0].get('prompt', '') if baseline_result.vlm_trace else ''
            raw_response = baseline_result.vlm_trace[0].get('response', '') if baseline_result.vlm_trace else ''
            
            baseline_results.append({
                "sample_index": i + 1,
                "image_path": sample["image_path"],
                "ground_truth": {"class": gt_class, "risk": gt_risk},
                "predicted": {"class": baseline_result.predicted_class},
                "correct": baseline_result.predicted_class == gt_class,
                "reasoning_trace": baseline_result.reasoning_trace,
                "prompt_given": prompt_given,
                "raw_response": raw_response
            })
            
            baseline_metrics = compute_metrics(
                baseline_preds, baseline_gt, baseline_states, baseline_risks
            )
            baseline_output_data = {
                "metadata": {"samples_processed": i + 1, "total_samples": len(dataset)},
                "live_metrics": {
                    "precision_macro": baseline_metrics.precision,
                    "recall_macro": baseline_metrics.recall,
                    "precision_normal": baseline_metrics.precision_normal,
                    "recall_normal": baseline_metrics.recall_normal,
                    "f1_score": baseline_metrics.f1_score,
                    "f1_macro": baseline_metrics.f1_macro,
                    "f1_confident": baseline_metrics.f1_confident,
                    "fnr_high_risk": baseline_metrics.fnr
                },
                "samples": baseline_results
            }
            with open(baseline_output, 'w') as f:
                json.dump(baseline_output_data, f, indent=2, default=str)
            
            print("NEURO-SYMBOLIC...")
            neuro_result = neuro_pipeline.run(sample["image_path"])
            neuro_preds.append(neuro_result.predicted_class)
            neuro_gt.append(gt_class)
            neuro_states.append(neuro_result.state)
            neuro_risks.append(gt_risk)
            
            neuro_results.append({
                "sample_index": i + 1,
                "image_path": sample["image_path"],
                "ground_truth": {"class": gt_class, "risk": gt_risk},
                "predicted": {"class": neuro_result.predicted_class},
                "correct": neuro_result.predicted_class == gt_class,
                "state": neuro_result.state.value,
                "vlm_attributes": neuro_result.attribute_confidences,
                "kg_energies": neuro_result.kg_energies,
                "psl_scores": neuro_result.psl_scores,
                "feedback_details": neuro_result.feedback_details,
                "reasoning_trace": neuro_result.reasoning_trace,
                "timing_ms": neuro_result.timing_ms
            })
            
            for key in timing_stats:
                if key in neuro_result.timing_ms:
                    timing_stats[key].append(neuro_result.timing_ms[key])
            
            neuro_metrics = compute_metrics(
                neuro_preds, neuro_gt, neuro_states, neuro_risks
            )
            neuro_output_data = {
                "metadata": {"samples_processed": i + 1, "total_samples": len(dataset)},
                "live_metrics": {
                    "precision_macro": neuro_metrics.precision,
                    "recall_macro": neuro_metrics.recall,
                    "precision_normal": neuro_metrics.precision_normal,
                    "recall_normal": neuro_metrics.recall_normal,
                    "f1_score": neuro_metrics.f1_score,
                    "f1_macro": neuro_metrics.f1_macro,
                    "f1_confident": neuro_metrics.f1_confident,
                    "hrr": neuro_metrics.hrr,
                    "fnr_high_risk": neuro_metrics.fnr
                },
                "samples": neuro_results
            }
            with open(neuro_output, 'w') as f:
                json.dump(neuro_output_data, f, indent=2, default=str)
            
            print(f"  Baseline: {baseline_result.predicted_class} | NeuroSymb: {neuro_result.predicted_class} | GT: {gt_class}")
        
        secondary_path = f"{output_base}_secondary_metrics.json"
        feedback_triggered = sum(1 for r in neuro_results if r.get('feedback_details', {}).get('triggered', False))
        feedback_iterations = [r.get('feedback_details', {}).get('iterations', 0) for r in neuro_results if r.get('feedback_details', {}).get('triggered', False)]

        num_classes = len(neuro_pipeline.kg.classes)
        num_attributes = len(getattr(neuro_pipeline.kg, "active_attributes", neuro_pipeline.kg.all_attributes))
        nodes = num_classes + num_attributes
        edges = neuro_pipeline.kg.count_edges()
        grounded = num_classes * num_attributes
        
        secondary_metrics = {
            "timing_statistics": {
                key: {"mean": float(np.mean(values)) if values else 0.0, "std": float(np.std(values)) if values else 0.0}
                for key, values in timing_stats.items() if values
            },
            "kg_complexity": {
                "theoretical": f"O(|Y| x |A|) = O({num_classes} x {num_attributes}) = O({grounded})",
                "nodes": nodes,
                "edges": edges,
                "grounded_facts_per_image": grounded,
                "measured_avg_ms": float(np.mean(timing_stats['kg_reasoning'])) if timing_stats['kg_reasoning'] else 0.0
            },
            "feedback_statistics": {"trigger_rate": feedback_triggered / len(neuro_results) if neuro_results else 0.0, "avg_iterations": float(np.mean(feedback_iterations)) if feedback_iterations else 0.0}
        }
        with open(secondary_path, 'w') as f:
            json.dump(secondary_metrics, f, indent=2)
        
        print("\n" + "=" * 30)
        print("FINAL COMPARISON RESULTS")
        print(f"{'Metric':<15} {'Baseline':>12} {'Neuro-Symb':>12}")
        print("-" * 30)
        print(f"{'Prec (Macro)':<15} {baseline_metrics.precision:>12.4f} {neuro_metrics.precision:>12.4f}")
        print(f"{'Prec (Normal)':<15} {baseline_metrics.precision_normal:>12.4f} {neuro_metrics.precision_normal:>12.4f}")
        print(f"{'Recall (Macro)':<15} {baseline_metrics.recall:>12.4f} {neuro_metrics.recall:>12.4f}")
        print(f"{'Recall (Normal)':<15} {baseline_metrics.recall_normal:>12.4f} {neuro_metrics.recall_normal:>12.4f}")
        print(f"{'F1-Score':<15} {baseline_metrics.f1_score:>12.4f} {neuro_metrics.f1_score:>12.4f}")
        print(f"{'F1-Macro':<15} {baseline_metrics.f1_macro:>12.4f} {neuro_metrics.f1_macro:>12.4f}")
        print(f"{'F1-Conf':<15} {baseline_metrics.f1_confident:>12.4f} {neuro_metrics.f1_confident:>12.4f}")
        print(f"{'HRR':<15} {'N/A':>12} {neuro_metrics.hrr*100:>11.2f}%")
        print(f"{'FNR (High)':<15} {baseline_metrics.fnr*100:>11.2f}% {neuro_metrics.fnr*100:>11.2f}%")
        print(f"\nSaved: {baseline_output}, {neuro_output}, {secondary_path}")
        
    elif args.baseline or args.baseline_attributes:
        print(f"Running {'BASELINE' if args.baseline else 'BASELINE + ATTRIBUTES'} evaluation...")
        
        verbose = True
        dataset = load_dataset(args.dataset)
        kg_path = "data/knowledge_graph.json"
        
        if args.baseline:
            from baseline import BaselinePipeline
            pipeline = BaselinePipeline(
                vlm_provider=args.provider,
                **vlm_kwargs
            )
        else:
            from baseline_attributes import BaselineAttributesPipeline
            pipeline = BaselineAttributesPipeline(
                kg_path=kg_path,
                vlm_provider=args.provider,
                **vlm_kwargs
            )
        
        baseline_output = args.output
        if baseline_output == "results.json":
             baseline_output = "results_baseline.json" if args.baseline else "results_baseline_attributes.json"
        
        existing_results, processed_paths = load_existing_results(baseline_output)
        all_results = list(existing_results)

        
        preds, gt, states, risks = [], [], [], []
        for existing in existing_results:
            if 'predicted' in existing and 'ground_truth' in existing:
                preds.append(existing['predicted'].get('class', 'UNKNOWN'))
                gt_info = existing.get('ground_truth', {})
                gt.append(gt_info.get('class', 'UNKNOWN'))
                states.append(SystemState.NORMAL)
                risks.append(gt_info.get('risk'))
        
        if processed_paths:
            print(f"[RESUME] Will skip {len(processed_paths)} already processed samples")
        
        current_metrics = None
        skipped_count = 0
        
        for run_idx in range(args.runs):
            print(f"Run {run_idx+1}/{args.runs}")
            
            for i, sample in enumerate(dataset):
                image_path = sample["image_path"]
                
                if image_path in processed_paths:
                    skipped_count += 1
                    if skipped_count <= 5 or skipped_count % 100 == 0:
                        print(f"  [SKIP] {image_path} (already processed)")
                    continue
                
                if verbose and i % 10 == 0:
                    print(f"  Processing {i}/{len(dataset)} (new: {len(all_results) - len(existing_results) + 1})...")

                result = pipeline.run(image_path)
                
                preds.append(result.predicted_class)
                gt.append(sample["class_name"])
                states.append(result.state)
                risks.append(sample.get("risk"))

                all_results.append({
                    "sample_index": len(all_results) + 1,
                    "image_path": image_path,
                    "ground_truth": {"class": sample["class_name"]},
                    "predicted": {"class": result.predicted_class},
                    "correct": result.predicted_class == sample["class_name"],
                    "reasoning_trace": result.reasoning_trace
                })
                
                current_metrics = compute_metrics(preds, gt, states, risks)
                
                output_data = {
                    "metadata": {
                        "samples_processed": len(all_results),
                        "total_samples": len(dataset),
                        "run": run_idx + 1
                    },
                    "live_metrics": {
                        "accuracy": current_metrics.accuracy,
                        "precision_macro": current_metrics.precision,
                        "recall_macro": current_metrics.recall,
                        "precision_normal": current_metrics.precision_normal,
                        "recall_normal": current_metrics.recall_normal,
                        "f1_score": current_metrics.f1_score,
                        "f1_macro": current_metrics.f1_macro,
                        "f1_confident": current_metrics.f1_confident,
 #                       "fnr": current_metrics.fnr,
                        "hrr": current_metrics.hrr,
                        "fnr_high_risk": current_metrics.fnr
                    },
                    "samples": all_results
                }
                
                with open(baseline_output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)

        print("\n" + "=" * 30)
        print("       BASELINE RESULTS")
        print("=" * 30)
        print(f"{'Metric':<15} {'Value':>10}")
        print(f"{'Prec (Macro)':<15} {current_metrics.precision:>10.4f}")
        print(f"{'Prec (Normal)':<15} {current_metrics.precision_normal:>10.4f}")
        print(f"{'Recall (Macro)':<15} {current_metrics.recall:>10.4f}")
        print(f"{'Recall (Normal)':<15} {current_metrics.recall_normal:>10.4f}")
        print(f"{'F1-Score':<15} {current_metrics.f1_score:>10.4f}")
        print(f"{'F1-Macro':<15} {current_metrics.f1_macro:>10.4f}")
        print(f"{'F1-Conf':<15} {current_metrics.f1_confident:>10.4f}")
        print(f"{'HRR':<15} {current_metrics.hrr*100:>9.2f}%")
        print(f"{'FNR (High)':<15} {current_metrics.fnr*100:>9.2f}%")
        print(f"\nBaseline results saved to {baseline_output}")
        
    else:
        run_evaluation(
            dataset_path=args.dataset,
            provider=args.provider,
            num_runs=args.runs,
            output_path=args.output,
            **vlm_kwargs
        )
