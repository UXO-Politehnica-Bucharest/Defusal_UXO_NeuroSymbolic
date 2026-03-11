"""
Hybrid Feedback Mechanism for Error Correction.

This module implements the closed-loop feedback mechanism that
detects label shifts and generates targeted queries to correct VLM errors.

Reference: Paper Section 4.3 - Hybrid Feedback Mechanism
"""
from typing import Dict, List, Optional

from core.types import (
    ConstraintViolation,
    FeedbackQuery,
    QueryType,
    PSLResult,
    AttributeConfidences,
)


class HybridFeedbackMechanism:
    """
    Implements the Hybrid Feedback Mechanism from the paper.
    
    Detects Constrained-Induced Label Shift (when ŷ_graph ≠ ŷ_psl)
    and generates targeted queries for the attribute most responsible
    for the inconsistency.
    """
    
    def __init__(
        self, 
        attribute_definitions: Dict[str, str],
        kg_classes: Optional[Dict] = None
    ):
        """
        Initialize the feedback mechanism.
        
        Args:
            attribute_definitions: Dictionary mapping attributes to their visual descriptions
            kg_classes: Optional Knowledge Graph class definitions (unused in paper implementation)
        """
        self.query_history: List[str] = []
        self.attribute_definitions = attribute_definitions
        self.kg_classes = kg_classes or {}
    
    def detect_label_shift(
        self,
        graph_hypothesis: str,
        psl_hypothesis: str
    ) -> bool:
        """
        Detect if there is a label shift between graph and PSL hypotheses.
        
        Label shift indicates the system is at a fragile decision boundary
        where visual evidence supports one conclusion but logical rules
        mandate another.
        
        Reference: Paper Equation (5)
        
        Args:
            graph_hypothesis: ŷ_graph from argmax S_graph(y|a)
            psl_hypothesis: ŷ_psl from argmin J_PSL(y, a)
        
        Returns:
            True if label shift detected
        """
        return graph_hypothesis != psl_hypothesis
    
    def identify_responsible_attribute(
        self,
        violations: List[ConstraintViolation]
    ) -> Optional[ConstraintViolation]:
        """
        Identify the attribute most responsible for the inconsistency.
        
        a* = argmax violation(a)
        
        Reference: Paper Equation (6)
        
        Args:
            violations: List of constraint violations from PSL
        
        Returns:
            The violation with maximum magnitude, or None if no violations
        """
        if not violations:
            return None
        active_violations = [v for v in violations if v.violation_magnitude > 0]
        if not active_violations:
            return None
        return max(active_violations, key=lambda v: v.violation_magnitude)
    
    def generate_query(
        self,
        violation: ConstraintViolation,
        previous_confidence: Optional[float] = None
    ) -> FeedbackQuery:
        """
        Generate a targeted query to verify an attribute status.
        
        Args:
            violation: The constraint violation to address
            previous_confidence: Previous confidence value (for context)
        
        Returns:
            FeedbackQuery with targeted prompt
        """
        attr_readable = violation.attribute.replace('_', ' ')
        prev_conf = previous_confidence if previous_confidence is not None else 0.0

        if violation.constraint_type == "necessity":
            prompt = (
                "Look very carefully at this image again.\n"
                f"Is there any '{attr_readable}' visible on this object?\n"
                "Consider:\n"
                "- It might be partially obscured by dirt, rust, or damage\n"
                "- Check all angles and surfaces visible\n"
                "- Look for even subtle evidence\n"
                "Respond with ONLY a JSON object: {\"confidence\": X.XX}\n"
                "where X.XX is your confidence between 0.0 and 1.0."
            )
            query_type = QueryType.COMPLETENESS
        else:
            prompt = (
                "Re-examine this image carefully.\n"
                f"You previously detected '{attr_readable}' with confidence {prev_conf:.2f}.\n"
                f"Please verify: Is this definitely '{attr_readable}', or could it be:\n"
                "- Visual noise or shadows?\n"
                "- Damage or corrosion mistaken for a feature?\n"
                "- A different but similar-looking component?\n"
                "Respond with ONLY a JSON object: {\"confidence\": X.XX}\n"
                "where X.XX is your revised confidence between 0.0 and 1.0."
            )
            query_type = QueryType.CONTRADICTION
        
        return FeedbackQuery(
            attribute=violation.attribute,
            query_type=query_type,
            prompt=prompt,
            violation_magnitude=violation.violation_magnitude
        )
    
    def generate_batch_queries(
        self,
        violations: List[ConstraintViolation]
    ) -> List[FeedbackQuery]:
        """
        Generate queries for ALL violations at once for batch processing.
        
        This is more efficient than single-attribute queries as it allows
        the VLM to re-evaluate multiple attributes in a single call.
        
        Args:
            violations: List of all constraint violations
        
        Returns:
            List of FeedbackQuery objects for all significant violations
        """
        queries = []
        for violation in violations:
            if violation.violation_magnitude > 0:
                queries.append(self.generate_query(violation))
        return queries
    
    def update_batch_confidences(
        self,
        current_confidences: AttributeConfidences,
        updates: Dict[str, float]
    ) -> AttributeConfidences:
        """
        Update multiple attribute confidences at once.
        
        Args:
            current_confidences: Current attribute confidence dict
            updates: Dict mapping attribute names to new confidence values
        
        Returns:
            Updated confidence dict
        """
        updated = current_confidences.copy()
        for attr, conf in updates.items():
            updated[attr] = conf
            self.query_history.append(attr)
        return updated
    
    def should_trigger_feedback(
        self,
        graph_hypothesis: str,
        psl_result: PSLResult,
        iteration: int,
        max_iterations: Optional[int] = None
    ) -> bool:
        """
        Determine if feedback should be triggered.
        
        Feedback is triggered when:
        1. There is a label shift (graph != psl)
        2. We haven't exceeded max iterations
        3. There are violations to address
        
        Args:
            graph_hypothesis: Initial graph hypothesis
            psl_result: PSL validation result
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
        Returns:
            True if feedback should be triggered
        """
        if max_iterations is not None and iteration >= max_iterations:
            return False
        
        is_label_shift = self.detect_label_shift(graph_hypothesis, psl_result.class_name)
        if not is_label_shift:
            return False
        return bool(psl_result.violations)
    
    def update_confidences(
        self,
        current_confidences: AttributeConfidences,
        attribute: str,
        new_confidence: float
    ) -> AttributeConfidences:
        """
        Update attribute confidences with new VLM response.
        
        Args:
            current_confidences: Current attribute confidence dict
            attribute: Attribute to update
            new_confidence: New confidence value from VLM
        
        Returns:
            Updated confidence dict
        """
        updated = current_confidences.copy()
        updated[attribute] = new_confidence
        self.query_history.append(attribute)
        return updated
    
    def get_feedback_summary(
        self,
        graph_hypothesis: str,
        psl_result: PSLResult
    ) -> Dict:
        """
        Get a summary of the feedback analysis.
        
        Useful for logging and debugging.
        """
        responsible = self.identify_responsible_attribute(psl_result.violations)
        
        return {
            "label_shift_detected": self.detect_label_shift(
                graph_hypothesis, psl_result.class_name
            ),
            "graph_hypothesis": graph_hypothesis,
            "psl_hypothesis": psl_result.class_name,
            "total_violations": len(psl_result.violations),
            "responsible_attribute": responsible.attribute if responsible else None,
            "attribute_group": responsible.attribute_group if responsible else None,
            "violation_type": responsible.constraint_type if responsible else None,
            "violation_magnitude": responsible.violation_magnitude if responsible else 0.0
        }
    
    def reset(self) -> None:
        """Reset the query history for a new inference."""
        self.query_history = []
