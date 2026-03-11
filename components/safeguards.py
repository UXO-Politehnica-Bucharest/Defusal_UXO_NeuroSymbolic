"""
Safeguards and Uncertainty Management.

This module implements safety mechanisms to prevent incorrect predictions
and escalate uncertain cases to human review.

Reference: Paper Section 4.4 - Safeguards and Uncertainty Management

Principle: False negative = catastrophic failure
          Manual review request = acceptable operational cost
"""
from typing import List, Optional

from core.types import (
    SystemState,
    SafeguardResult,
    PSLResult,
)


class Safeguards:
    """
    Implements safeguards for the neuro-symbolic pipeline.
    
    Ensures fail-safe operation by detecting:
    1. High missing evidence in the initial hypothesis
    2. Persistent label shift after feedback iteration
    """
    
    def __init__(
        self,
        max_iterations: int = 2
    ):
        """
        Initialize safeguards.
        
        Args:
            max_iterations: Maximum feedback iterations before escalation
        """
        self.max_iterations = max_iterations
    
    def check_initial_hypothesis(
        self,
        min_energy: float,
        is_uncertain: bool = False
    ) -> SafeguardResult:
        """
        Check if the initial hypothesis is valid.
        
        Args:
            min_energy: Minimum KG energy from initial hypothesis
            is_uncertain: True if energy exceeds required attribute count
        
        Returns:
            SafeguardResult
        """
        if is_uncertain:
            return SafeguardResult(
                state=SystemState.UNCERTAINTY,
                reason=(
                    "High missing evidence: min J_KG exceeds required attribute count."
                ),
                can_proceed=False
            )
        
        return SafeguardResult(
            state=SystemState.NORMAL,
            reason=None,
            can_proceed=True
        )

    def check_persistent_label_shift(
        self,
        iteration: int,
        graph_hypothesis: str,
        psl_hypotheses_history: List[str],
        max_iterations: Optional[int] = None
    ) -> SafeguardResult:
        """
        Check for persistent label shift after multiple iterations.
        
        If ŷ_graph ≠ ŷ_psl^(1) ≠ ŷ_psl^(2), the inconsistency indicates
        irresolvable visual ambiguity and should be escalated to human review.
        
        Reference: Paper - "Persistent Label Shift"
        
        Args:
            iteration: Current iteration number
            graph_hypothesis: Initial graph hypothesis
            psl_hypotheses_history: List of PSL hypotheses across iterations
        
        Returns:
            SafeguardResult indicating if we should stop
        """
        limit = self.max_iterations if max_iterations is None else max_iterations
        if limit is None:
            return SafeguardResult(
                state=SystemState.NORMAL,
                reason=None,
                can_proceed=True
            )

        if len(psl_hypotheses_history) >= 2 and iteration >= limit:
            psl_first = psl_hypotheses_history[0]
            psl_second = psl_hypotheses_history[1]

            if (
                graph_hypothesis != psl_first
                and psl_first != psl_second
                and graph_hypothesis != psl_second
            ):
                return SafeguardResult(
                    state=SystemState.UNCERTAINTY,
                    reason=(
                        f"Persistent label shift after {iteration} iterations. "
                        f"Graph hypothesis: {graph_hypothesis}, "
                        f"PSL hypotheses: {psl_hypotheses_history}. "
                        "Irresolvable ambiguity - escalating to human review."
                    ),
                    can_proceed=False
                )
        
        return SafeguardResult(
            state=SystemState.NORMAL,
            reason=None,
            can_proceed=True
        )
    
    
    def run_all_checks(
        self,
        graph_score: float,
        psl_result: Optional[PSLResult] = None,
        iteration: int = 0,
        psl_hypotheses_history: Optional[List[str]] = None,
        graph_hypothesis: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> SafeguardResult:
        """
        Run all safeguard checks and return first failure.
        
        Args:
            graph_score: Best graph score
            psl_result: Optional PSL result
            iteration: Current iteration
            psl_hypotheses_history: History of PSL hypotheses
            graph_hypothesis: Initial graph hypothesis
        
        Returns:
            First failing SafeguardResult, or success if all pass
        """
        check = self.check_initial_hypothesis(graph_score)
        if not check.can_proceed:
            return check
        
        if psl_hypotheses_history and graph_hypothesis:
            check = self.check_persistent_label_shift(
                iteration, graph_hypothesis, psl_hypotheses_history, max_iterations=max_iterations
            )
            if not check.can_proceed:
                return check
        
        return SafeguardResult(
            state=SystemState.NORMAL,
            reason=None,
            can_proceed=True
        )
