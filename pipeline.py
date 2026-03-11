"""
Neuro-Symbolic Pipeline for UXO Recognition.

This is the main pipeline that integrates all components:
1. VLM Inspector for attribute extraction
2. Knowledge Graph for class hypothesis
3. PSL Validator for consistency checking
4. Hybrid Feedback for error correction
5. Safeguards for uncertainty management

Reference: Paper Section 4 - Methods, Figure 2
"""
import time
from typing import List, Optional

from core.types import (
    SystemState,
    InferenceResult,
    AttributeConfidences,
    QueryType,
)
from components.knowledge_graph import KnowledgeGraphParser
from components.psl_validator import PSLValidator
from components.hybrid_feedback import HybridFeedbackMechanism
from components.safeguards import Safeguards
from components.vlm_inspector import VLMInspector, create_vlm_inspector


class NeuroSymbolicPipeline:
    """
    Main pipeline implementing the closed-loop neuro-symbolic framework.
    
    The pipeline follows the Perceive → Reason → Correct loop:
    1. Extract attributes using VLM
    2. Compute graph hypothesis from KG
    3. Validate consistency with PSL
    4. If label shift detected, trigger feedback and re-query
    5. Apply safeguards and return final result
    """
    
    def __init__(
        self,
        kg_path: str = "data/knowledge_graph.json",
        vlm_provider: str = "LocalVLLMProvider",
        **vlm_kwargs
    ):
        """
        Initialize the pipeline.
        
        Args:
            kg_path: Path to the Knowledge Graph JSON
            vlm_provider: VLM provider type ("openai", "local", "transformer", "mock")
            **vlm_kwargs: Additional arguments for VLM provider
        """
        self.kg = KnowledgeGraphParser(kg_path)
        self.psl = PSLValidator(self.kg)
        self.feedback = HybridFeedbackMechanism(
            attribute_definitions=self.kg.attribute_definitions,
            kg_classes=self.kg.classes
        )
        self.safeguards = Safeguards()
        
        self.vlm = create_vlm_inspector(
            provider_type=vlm_provider,
            attributes_list=self.kg.active_attributes,
            attribute_definitions=self.kg.attribute_definitions,
            contradictory_pairs=self.kg.contradictory_pairs,
            **vlm_kwargs
        )
        
    
    def run(self, image_path: str) -> InferenceResult:
        """
        Run the full inference pipeline on an image.
        
        Args:
            image_path: Path to the ordnance image
        
        Returns:
            InferenceResult with class, confidence, and trace
        """
        trace = []
        timing = {'total': 0, 'vlm_inference': 0, 'kg_reasoning': 0, 'psl_inference': 0, 'feedback_loop': 0}
        feedback_details = {'triggered': False, 'iterations': 0, 're_queries': []}
        t_start = time.time()
        
        self.feedback.reset()
        if hasattr(self.vlm, 'clear_history'):
            self.vlm.clear_history()
        
        trace.append(f"Step 1: Extracting visual attributes from image")
        t_vlm_start = time.time()
        attr_conf = self.vlm.extract_attributes(image_path)
        timing['vlm_inference'] = (time.time() - t_vlm_start) * 1000
        detected = sum(1 for v in attr_conf.values() if v > 1e-6)
        trace.append(f"  Detected {detected} attributes above 0.0 confidence")
        
        if getattr(self.vlm, 'extraction_failed', False):
            trace.append(f"  SAFEGUARD: {self.vlm.failure_reason}")
            timing['total'] = (time.time() - t_start) * 1000
            return InferenceResult(
                predicted_class="UNCERTAIN",
                confidence=0.0,
                state=SystemState.UNCERTAINTY,
                reasoning_trace=trace,
                iterations_used=0,
                attribute_confidences=attr_conf,
                vlm_trace=getattr(self.vlm, 'history', []),
                kg_energies={},
                psl_scores={},
                feedback_details=feedback_details,
                timing_ms=timing
            )

        trace.append("Step 2: Computing graph hypothesis (J_KG energy)")
        t_kg_start = time.time()
        kg_energies = self.kg.compute_all_kg_energies(attr_conf)
        graph_hyp, min_energy = self.kg.get_initial_hypothesis(attr_conf)
        timing['kg_reasoning'] = (time.time() - t_kg_start) * 1000
        trace.append(f"  ŷ_graph = {graph_hyp} (energy: {min_energy:.3f})")
        
        is_uncertain = self.kg.is_uncertain(graph_hyp, attr_conf)
        if is_uncertain:
            trace.append("  SAFEGUARD: High missing evidence indicates uncertainty!")
        
        sg_check = self.safeguards.check_initial_hypothesis(
            min_energy, 
            is_uncertain=is_uncertain
        )
        if not sg_check.can_proceed:
            trace.append(f"  SAFEGUARD: {sg_check.reason}")
            timing['total'] = (time.time() - t_start) * 1000
            return InferenceResult(
                predicted_class="UNCERTAIN",
                confidence=0.0,
                state=SystemState.UNCERTAINTY,
                reasoning_trace=trace,
                iterations_used=0,
                attribute_confidences=attr_conf,
                vlm_trace=getattr(self.vlm, 'history', []),
                kg_energies=kg_energies,
                psl_scores={},
                feedback_details=feedback_details,
                timing_ms=timing
            )
        
        trace.append("Step 3: PSL consistency validation")
        t_psl_start = time.time()
        psl_result = self.psl.find_minimum_energy_class(attr_conf, kg_energies)
        timing['psl_inference'] = (time.time() - t_psl_start) * 1000
        trace.append(
            f"  ŷ_psl = {psl_result.class_name} "
            f"(class_energy: {psl_result.total_energy:.3f})"
        )
        if psl_result.objective_energy is not None:
            trace.append(
                f"  PSL objective energy: {psl_result.objective_energy:.3f}"
            )
        
        psl_hypotheses = [psl_result.class_name]
        graph_hyp_initial = graph_hyp
        iteration = 0
        max_iterations = self.safeguards.max_iterations
        
        t_feedback_start = time.time()
        while self.feedback.should_trigger_feedback(
            graph_hyp, psl_result, iteration, max_iterations
        ):
            feedback_details['triggered'] = True
            iteration += 1
            feedback_details['iterations'] = iteration
            trace.append(f"Step 4.{iteration}: Label shift detected → feedback loop")
            
            summary = self.feedback.get_feedback_summary(graph_hyp, psl_result)
            trace.append(
                f"  Responsible attribute: {summary['responsible_attribute']} "
                f"({summary['violation_type']}, magnitude: {summary['violation_magnitude']:.3f})"
            )
            if summary.get("attribute_group"):
                group = " OR ".join(summary["attribute_group"])
                trace.append(f"  Attribute group: {group}")
            
            responsible = self.feedback.identify_responsible_attribute(
                psl_result.violations
            )
            
            if responsible:
                old_conf = attr_conf.get(responsible.attribute, 0.0)
                
                query = self.feedback.generate_query(
                    responsible,
                    previous_confidence=old_conf
                )
                trace.append(f"  Query: {query.query_type.value} for '{responsible.attribute}'")
                new_conf = self.vlm.query_specific_attribute(
                    image_path,
                    responsible.attribute,
                    query.query_type,
                    old_conf,
                    override_prompt=query.prompt
                )
                
                attr_conf = self.feedback.update_confidences(
                    attr_conf, responsible.attribute, new_conf
                )
                trace.append(f"  Updated {responsible.attribute}: {old_conf:.2f} → {new_conf:.2f}")
                feedback_details['re_queries'].append({
                    'attribute': responsible.attribute,
                    'old_confidence': old_conf,
                    'new_confidence': new_conf,
                    'query_type': query.query_type.value
                })

                old_rounded = round(old_conf, 2)
                new_rounded = round(new_conf, 2)
                if old_rounded == new_rounded:
                    trace.append(f"  (no meaningful change: {old_rounded} → {new_rounded})")
            else:
                trace.append("  No more attributes to query")
                break
            
            # Recalculate y_graph and y_psl2 AFTER feedback
            graph_hyp, min_energy = self.kg.get_initial_hypothesis(attr_conf)
            kg_energies = self.kg.compute_all_kg_energies(attr_conf)
            psl_result = self.psl.find_minimum_energy_class(attr_conf, kg_energies)
            psl_hypotheses.append(psl_result.class_name)
            
            trace.append(
                f"  After feedback: ŷ_graph={graph_hyp}, ŷ_psl={psl_result.class_name}"
            )

            sg_check = self.safeguards.check_persistent_label_shift(
                iteration, graph_hyp_initial, psl_hypotheses, max_iterations=max_iterations
            )
            if not sg_check.can_proceed:
                trace.append(f"  SAFEGUARD: {sg_check.reason}")
                timing['feedback_loop'] = (time.time() - t_feedback_start) * 1000
                timing['total'] = (time.time() - t_start) * 1000
                psl_scores = psl_result.optimized_posteriors if psl_result.optimized_posteriors else {}
                return InferenceResult(
                    predicted_class="UNCERTAIN",
                    confidence=0.0,
                    state=SystemState.UNCERTAINTY,
                    reasoning_trace=trace,
                    iterations_used=iteration,
                    attribute_confidences=attr_conf,
                    vlm_trace=getattr(self.vlm, 'history', []),
                    kg_energies=kg_energies,
                    psl_scores=psl_scores,
                    feedback_details=feedback_details,
                    timing_ms=timing
                )

            # CONSISTENCY CHECK (AFTER recalculating y_psl2):
            if psl_result.class_name == graph_hyp:
                trace.append(
                    f"  CONSISTENCY: y_psl2={psl_result.class_name} matches graph hypothesis"
                )
                break
        
        timing['feedback_loop'] = (time.time() - t_feedback_start) * 1000
        
        final_class = psl_result.class_name
        if final_class == "UNCERTAIN":
            trace.append("  SAFEGUARD: PSL returned UNCERTAIN")
            timing['total'] = (time.time() - t_start) * 1000
            psl_scores = psl_result.optimized_posteriors if psl_result.optimized_posteriors else {}
            return InferenceResult(
                predicted_class="UNCERTAIN",
                confidence=0.0,
                state=SystemState.UNCERTAINTY,
                reasoning_trace=trace,
                iterations_used=iteration,
                attribute_confidences=attr_conf,
                vlm_trace=getattr(self.vlm, 'history', []),
                kg_energies=kg_energies,
                psl_scores=psl_scores,
                feedback_details=feedback_details,
                timing_ms=timing
            )

        cls = self.kg.get_class(final_class)
        required_count = len(cls.required_attributes) if cls else 0
        if psl_result.total_energy > float(required_count):
            trace.append(
                f"  SAFEGUARD: PSL energy {psl_result.total_energy:.3f} exceeds "
                f"required count ({required_count})"
            )
            timing['total'] = (time.time() - t_start) * 1000
            psl_scores = psl_result.optimized_posteriors if psl_result.optimized_posteriors else {}
            return InferenceResult(
                predicted_class="UNCERTAIN",
                confidence=0.0,
                state=SystemState.UNCERTAINTY,
                reasoning_trace=trace,
                iterations_used=iteration,
                attribute_confidences=attr_conf,
                vlm_trace=getattr(self.vlm, 'history', []),
                kg_energies=kg_energies,
                psl_scores=psl_scores,
                feedback_details=feedback_details,
                timing_ms=timing
            )

        req_sum = 0.0
        forb_sum = 0.0
        if cls:
            for attr_spec in cls.required_attributes:
                if isinstance(attr_spec, (list, tuple)):
                    req_sum += max(attr_conf.get(a, 0.0) for a in attr_spec)
                else:
                    req_sum += attr_conf.get(attr_spec, 0.0)
            for attr_spec in cls.forbidden_attributes:
                if isinstance(attr_spec, (list, tuple)):
                    forb_sum += max(attr_conf.get(a, 0.0) for a in attr_spec)
                else:
                    forb_sum += attr_conf.get(attr_spec, 0.0)

        if req_sum <= forb_sum:
            trace.append(
                f"  SAFEGUARD: Required evidence ({req_sum:.3f}) "
                f"does not exceed forbidden evidence ({forb_sum:.3f})"
            )
            timing['total'] = (time.time() - t_start) * 1000
            psl_scores = psl_result.optimized_posteriors if psl_result.optimized_posteriors else {}
            return InferenceResult(
                predicted_class="UNCERTAIN",
                confidence=0.0,
                state=SystemState.UNCERTAINTY,
                reasoning_trace=trace,
                iterations_used=iteration,
                attribute_confidences=attr_conf,
                vlm_trace=getattr(self.vlm, 'history', []),
                kg_energies=kg_energies,
                psl_scores=psl_scores,
                feedback_details=feedback_details,
                timing_ms=timing
            )
        confidence = 0.0
        if psl_result.optimized_posteriors:
            confidence = max(0.0, psl_result.optimized_posteriors.get(final_class, 0.0))
        timing['total'] = (time.time() - t_start) * 1000
        
        psl_scores = psl_result.optimized_posteriors if psl_result.optimized_posteriors else {psl_result.class_name: confidence}
        
        return InferenceResult(
            predicted_class=final_class,
            confidence=confidence,
            state=SystemState.NORMAL,
            reasoning_trace=trace,
            iterations_used=iteration,
            attribute_confidences=attr_conf,
            vlm_trace=getattr(self.vlm, 'history', []),
            kg_energies=kg_energies,
            psl_scores=psl_scores,
            feedback_details=feedback_details,
            timing_ms=timing
        )
    
    def run_batch(
        self, 
        image_paths: List[str],
        verbose: bool = False
    ) -> List[InferenceResult]:
        """
        Run inference on a batch of images.
        
        Args:
            image_paths: List of image paths
            verbose: Whether to print progress
        
        Returns:
            List of InferenceResults
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if verbose and i % 100 == 0:
                print(f"Processing {i}/{total}...")
            
            try:
                result = self.run(path)
                results.append(result)
            except Exception as e:
                results.append(InferenceResult(
                    predicted_class="ERROR",
                    confidence=0.0,
                    state=SystemState.UNCERTAINTY,
                    reasoning_trace=[f"Error: {str(e)}"],
                    iterations_used=0,
                    attribute_confidences={},
                    vlm_trace=[]
                ))
        
        return results


def demo_inference(image_path: str = None):
    """
    Demo function to test the pipeline.
    
    Args:
        image_path: Optional path to test image
    """
    print("=" * 60)
    print("  Neuro-Symbolic UXO Framework - Demo")
    print("=" * 60)
    
    pipeline = NeuroSymbolicPipeline(
        vlm_provider="mock"
    )
    
    result = pipeline.run(image_path or "test_image.jpg")
    
    print("\n--- Reasoning Trace ---")
    for line in result.reasoning_trace:
        print(line)
    
    print("\n--- Final Result ---")
    print(f"Predicted Class: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"System State: {result.state.value}")
    print(f"Iterations Used: {result.iterations_used}")
    
    return result


if __name__ == "__main__":
    demo_inference()
