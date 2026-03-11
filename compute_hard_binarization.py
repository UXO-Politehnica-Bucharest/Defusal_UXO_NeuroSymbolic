"""
Recompute metrics with FULL pipeline simulation.

Simulates the complete neuro-symbolic pipeline (KG → Safeguards → PSL → 
Feedback Loop → Persistent Label Shift) for both Hard Binarization and 
Defusal, using stored VLM attributes and feedback data from results JSON.

This produces the correct Table 2 values including HRR and FNR.

Usage:
    python recompute_hard_binarization.py -r results/qwen3_neurosymbolic.json
    python recompute_hard_binarization.py -r results/qwen3_neurosymbolic.json --limit 100
"""
import argparse
import json
import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

\
import importlib.util

def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_base = os.path.dirname(os.path.abspath(__file__))
_kg_mod = _import_from_file("knowledge_graph", os.path.join(_base, "components", "knowledge_graph.py"))
_psl_mod = _import_from_file("psl_validator", os.path.join(_base, "components", "psl_validator.py"))
_fb_mod = _import_from_file("hybrid_feedback", os.path.join(_base, "components", "hybrid_feedback.py"))
_sg_mod = _import_from_file("safeguards", os.path.join(_base, "components", "safeguards.py"))

KnowledgeGraphParser = _kg_mod.KnowledgeGraphParser
PSLValidator = _psl_mod.PSLValidator
HybridFeedbackMechanism = _fb_mod.HybridFeedbackMechanism
Safeguards = _sg_mod.Safeguards

from core.types import SystemState
from evaluate import compute_metrics


@dataclass
class SampleData:
    """Minimal sample data for recomputation."""
    ground_truth: str
    risk: str
    vlm_attributes: Dict[str, float]
    original_prediction: str
    original_state: str
    feedback_details: Dict  # Contains re_queries with old/new confidences


def stream_samples(results_path: str, limit: Optional[int] = None):
    """Stream samples from a large results JSON."""
    count = 0
    current_sample = []
    brace_depth = 0
    in_samples = False

    with open(results_path, 'r') as f:
        for line in f:
            if '"samples"' in line and not in_samples:
                in_samples = True
                continue

            if not in_samples:
                continue

            for ch in line:
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1

            current_sample.append(line)

            if brace_depth == 0 and len(current_sample) > 1:
                sample_text = ''.join(current_sample).strip().rstrip(',')
                try:
                    sample = json.loads(sample_text)
                    if 'vlm_attributes' in sample and 'ground_truth' in sample:
                        yield SampleData(
                            ground_truth=sample['ground_truth']['class'],
                            risk=sample['ground_truth'].get('risk', 'High'),
                            vlm_attributes=sample['vlm_attributes'],
                            original_prediction=sample.get('predicted', {}).get('class', 'UNKNOWN'),
                            original_state=sample.get('state', 'normal'),
                            feedback_details=sample.get('feedback_details', {}),
                        )
                        count += 1
                        if limit and count >= limit:
                            return
                except (json.JSONDecodeError, KeyError):
                    pass
                current_sample = []
                brace_depth = 0


def simulate_full_pipeline(
    attr_conf: Dict[str, float],
    kg: KnowledgeGraphParser,
    psl: PSLValidator,
    feedback: HybridFeedbackMechanism,
    safeguards: Safeguards,
    stored_feedback: Dict,
    binarize: bool = False,
    threshold: float = 0.5,
) -> Tuple[str, SystemState]:
    """
    Simulate the full neuro-symbolic pipeline on given attributes.
    
    Steps:
    1. (Optionally) binarize attributes
    2. KG hypothesis + initial uncertainty safeguard
    3. PSL validation
    4. Feedback loop (using stored feedback data for re-queries)
    5. Persistent label shift safeguard
    
    Args:
        attr_conf: Initial VLM attribute confidences
        kg: KnowledgeGraphParser
        psl: PSLValidator
        feedback: HybridFeedbackMechanism
        safeguards: Safeguards
        stored_feedback: Stored feedback_details from original run
        binarize: Whether to binarize attributes (Hard Binarization mode)
        threshold: Binarization threshold
    
    Returns:
        (predicted_class, state)
    """
   
    if binarize:
        attr_conf = {k: (1.0 if v >= threshold else 0.0) for k, v in attr_conf.items()}
    

    graph_hyp, min_energy = kg.get_initial_hypothesis(attr_conf)
    

    is_uncertain = kg.is_uncertain(graph_hyp, attr_conf)
    sg_check = safeguards.check_initial_hypothesis(min_energy, is_uncertain=is_uncertain)
    if not sg_check.can_proceed:
        return "UNCERTAIN", SystemState.UNCERTAINTY
    
    
    kg_energies = kg.compute_all_kg_energies(attr_conf)
    psl_result = psl.find_minimum_energy_class(attr_conf, kg_energies)
    
    
    psl_hypotheses = [psl_result.class_name]
    graph_hyp_initial = graph_hyp
    iteration = 0
    max_iterations = safeguards.max_iterations
    
   
    re_queries = stored_feedback.get('re_queries', [])
    
    while feedback.should_trigger_feedback(graph_hyp, psl_result, iteration, max_iterations):
        iteration += 1
        
        # Try to use stored feedback data
        if iteration - 1 < len(re_queries):
            rq = re_queries[iteration - 1]
            attr_name = rq.get('attribute', '')
            new_conf = rq.get('new_confidence', 0.0)
            
            if binarize:
                new_conf = 1.0 if new_conf >= threshold else 0.0
            
            if attr_name and attr_name in attr_conf:
                attr_conf = dict(attr_conf)
                attr_conf[attr_name] = new_conf
        else:
            # No stored feedback for this iteration, try identifying responsible attribute
            responsible = feedback.identify_responsible_attribute(psl_result.violations)
            if not responsible:
                break
            
            break
        
        
        if binarize:
            attr_conf = {k: (1.0 if v >= threshold else 0.0) for k, v in attr_conf.items()}
        
        graph_hyp, min_energy = kg.get_initial_hypothesis(attr_conf)
        kg_energies = kg.compute_all_kg_energies(attr_conf)
        psl_result = psl.find_minimum_energy_class(attr_conf, kg_energies)
        psl_hypotheses.append(psl_result.class_name)
        
    
        sg_check = safeguards.check_persistent_label_shift(
            iteration, graph_hyp_initial, psl_hypotheses, max_iterations=max_iterations
        )
        if not sg_check.can_proceed:
            return "UNCERTAIN", SystemState.UNCERTAINTY
        
        # Convergence check
        if psl_result.class_name == graph_hyp:
            break
    
    final_class = psl_result.class_name
    if final_class == "UNCERTAIN":
        return "UNCERTAIN", SystemState.UNCERTAINTY
    
    return final_class, SystemState.NORMAL


def main():
    parser = argparse.ArgumentParser(
        description="Recompute Table 2 with full pipeline simulation"
    )
    parser.add_argument(
        "--results", "-r", type=str, required=True,
        help="Path to results JSON"
    )
    parser.add_argument(
        "--kg", "-k", type=str, default="data/knowledge_graph.json",
        help="Path to Knowledge Graph JSON"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="Binarization threshold (default: 0.5)"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit number of samples"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Table 2: Full Pipeline Recomputation")
    print("  (Hard Binarization vs Defusal with feedback + safeguards)")
    print("=" * 70)

    # Initialize components
    kg = KnowledgeGraphParser(args.kg)
    psl = PSLValidator(kg)
    feedback = HybridFeedbackMechanism(
        attribute_definitions=kg.attribute_definitions,
        kg_classes=kg.classes
    )
    safeguards = Safeguards()  # max_iterations=2

    print(f"\nKG: {len(kg.classes)} classes, {len(kg.active_attributes)} attributes")
    print(f"Safeguards max_iterations: {safeguards.max_iterations}")
    print(f"Results: {args.results}")
    if args.limit:
        print(f"Limit: {args.limit} samples")
    print(f"Binarization threshold: {args.threshold}")
    print()

    # Collect results for 3 conditions
    results = {
        'hard_bin': {'preds': [], 'gt': [], 'states': [], 'risks': []},
        'defusal': {'preds': [], 'gt': [], 'states': [], 'risks': []},
        'original': {'preds': [], 'gt': [], 'states': [], 'risks': []},
    }

    count = 0
    for sample in stream_samples(args.results, limit=args.limit):
        count += 1
        if count % 500 == 0 or count == 1:
            print(f"  Processing sample {count}...")

        # Check if original pipeline deferred this sample
        original_was_uncertain = (sample.original_state == 'uncertainty')

        if original_was_uncertain:
            # Preserve original UNCERTAIN — don't re-evaluate
            hb_pred, hb_state = "UNCERTAIN", SystemState.UNCERTAINTY
            df_pred, df_state = "UNCERTAIN", SystemState.UNCERTAINTY
        else:
            # 1. Hard Binarization — full pipeline with binarized attributes
            hb_pred, hb_state = simulate_full_pipeline(
                dict(sample.vlm_attributes), kg, psl, feedback, safeguards,
                sample.feedback_details, binarize=True, threshold=args.threshold
            )

            # 2. Defusal — full pipeline with soft attributes
            df_pred, df_state = simulate_full_pipeline(
                dict(sample.vlm_attributes), kg, psl, feedback, safeguards,
                sample.feedback_details, binarize=False
            )

        results['hard_bin']['preds'].append(hb_pred)
        results['hard_bin']['gt'].append(sample.ground_truth)
        results['hard_bin']['states'].append(hb_state)
        results['hard_bin']['risks'].append(sample.risk)

        results['defusal']['preds'].append(df_pred)
        results['defusal']['gt'].append(sample.ground_truth)
        results['defusal']['states'].append(df_state)
        results['defusal']['risks'].append(sample.risk)

        results['original']['preds'].append(sample.original_prediction)
        results['original']['gt'].append(sample.ground_truth)
        results['original']['states'].append(
            SystemState.UNCERTAINTY if original_was_uncertain
            else SystemState.NORMAL
        )
        results['original']['risks'].append(sample.risk)

    print(f"\nProcessed {count} samples total.\n")

  
    metrics = {}
    for key in ['hard_bin', 'defusal', 'original']:
        r = results[key]
        metrics[key] = compute_metrics(r['preds'], r['gt'], r['states'], r['risks'])

    # Print table
    hb = metrics['hard_bin']
    df = metrics['defusal']
    og = metrics['original']

    print("=" * 90)
    print("  RESULTS — Table 2: Neuro-Symbolic Engine Comparison (Full Pipeline)")
    print("=" * 90)
    header = f"{'Method':<28} {'F1-Score':>10} {'Recall':>10} {'F1-Conf':>10} {'HRR':>10} {'FNR':>10} {'Accuracy':>10}"
    print(header)
    print("-" * 90)
    
    for label, m in [("Hard Binarization", hb), ("Defusal / Łukasiewicz (Ours)", df)]:
        print(
            f"{label:<28} "
            f"{m.f1_score:>10.4f} "
            f"{m.recall:>10.4f} "
            f"{m.f1_confident:>10.4f} "
            f"{m.hrr*100:>9.1f}% "
            f"{m.fnr*100:>9.1f}% "
            f"{m.accuracy:>10.4f}"
        )
    print("-" * 90)
    print(
        f"{'Original Pipeline (stored)':<28} "
        f"{og.f1_score:>10.4f} "
        f"{og.recall:>10.4f} "
        f"{og.f1_confident:>10.4f} "
        f"{og.hrr*100:>9.1f}% "
        f"{og.fnr*100:>9.1f}% "
        f"{og.accuracy:>10.4f}"
    )
    print()

    print(f"{'Method':<28} {'Prec(M)':>10} {'F1-Macro':>10}")
    print("-" * 50)
    for label, m in [("Hard Binarization", hb), ("Defusal (Ours)", df), ("Original (stored)", og)]:
        print(f"{label:<28} {m.precision:>10.4f} {m.f1_macro:>10.4f}")


    for label, key in [("Hard Bin", 'hard_bin'), ("Defusal", 'defusal'), ("Original", 'original')]:
        r = results[key]
        n_uncertain = sum(1 for s in r['states'] if s == SystemState.UNCERTAINTY)
        n_normal = sum(1 for s in r['states'] if s == SystemState.NORMAL)
        print(f"\n{label}: {n_normal} NORMAL, {n_uncertain} UNCERTAIN ({n_uncertain/count*100:.1f}% deferred)")

    # Save output
    if args.output:
        output_data = {}
        for key, label in [('hard_bin', 'hard_binarization'), ('defusal', 'defusal_lukasiewicz'), ('original', 'original_pipeline')]:
            m = metrics[key]
            r = results[key]
            n_uncertain = sum(1 for s in r['states'] if s == SystemState.UNCERTAINTY)
            output_data[label] = {
                "f1_score": round(m.f1_score, 4),
                "recall": round(m.recall, 4),
                "f1_confident": round(m.f1_confident, 4),
                "hrr": round(m.hrr, 4),
                "fnr": round(m.fnr, 4),
                "accuracy": round(m.accuracy, 4),
                "precision_macro": round(m.precision, 4),
                "f1_macro": round(m.f1_macro, 4),
                "samples_normal": count - n_uncertain,
                "samples_uncertain": n_uncertain,
            }
        output_data["config"] = {
            "threshold": args.threshold,
            "max_iterations": safeguards.max_iterations,
            "samples_processed": count,
            "results_file": args.results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
