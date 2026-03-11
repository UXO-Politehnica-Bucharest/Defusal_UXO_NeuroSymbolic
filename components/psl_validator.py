"""
PSL Consistency Validator using Hinge-Loss Markov Random Field.

This module implements Probabilistic Soft Logic (PSL) validation
using Lukasiewicz operators to compute inconsistency energy.

Reference: Paper Section 4.2 - Consistency Validator, Equations (3) and (4)

The HL-MRF optimization finds optimal P(y) values that minimize the total
inconsistency energy across all constraints, as defined by:

    P(L|O) = (1/Z) exp(-Σ φ_j(L,O))     [Eq. 3 - psl_pdf]

where φ_j are convex hinge-loss potentials.
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize

from core.types import (
    ConstraintViolation,
    PSLResult,
    AttributeConfidences,
    GraphScores,
)
from core.lukasiewicz import (
    distance_to_satisfaction_implication,
    distance_to_satisfaction_exclusion,
)
from components.knowledge_graph import KnowledgeGraphParser


class PSLValidator:
    """
    Validates consistency using Probabilistic Soft Logic.
    
    Uses Hinge-Loss MRF to compute the "Inconsistency Energy" J_PSL
    which quantifies the distance to satisfaction of logical constraints.
    
    The energy function aggregates weighted violations:
    - Necessity constraints: P(y) → I(a) for required attributes
    - Exclusion constraints: P(y) → ¬I(a) for forbidden attributes
    """
    
    def __init__(self, kg: KnowledgeGraphParser, rule_weight: float = 1.0):
        """
        Initialize the PSL validator.
        
        Args:
            kg: Knowledge Graph parser instance
            rule_weight: Weight for all rules (default 1.0 for uniform weights)
        """
        self.kg = kg
        self.rule_weight = rule_weight
    
    def compute_class_energy(
        self,
        class_name: str,
        class_posterior: float,
        attribute_confidences: AttributeConfidences
    ) -> Tuple[float, List[ConstraintViolation]]:
        """
        Compute PSL energy for a specific class.
        
        J_PSL(y, a) = Σ max(0, P(y) - I(a_i)) for a_i ∈ N_req(y)
                    + Σ max(0, P(y) + I(a_j) - 1) for a_j ∈ N_forbid(y)
        
        Reference: Paper Equation (4)
        
        Args:
            class_name: Name of the ordnance class
            class_posterior: Posterior probability P(y) for this class
            attribute_confidences: VLM-derived attribute confidences
        
        Returns:
            Tuple of (total_energy, list of constraint violations)
        """
        violations = []
        # Strict Paper Logic (Lukasiewicz Sum) consistency
        # Eq 118: J_PSL = Sum(max(0, P(y) - I(req))) + Sum(max(0, P(y) + I(forbid) - 1))
        
        total_energy = 0.0
        
        cls = self.kg.get_class(class_name)
        if cls is None:
            return float('inf'), []

        req_groups = list(cls.required_attributes)
        forb_groups = list(cls.forbidden_attributes)
        
        for attr_spec in req_groups:
       
            if isinstance(attr_spec, list) or isinstance(attr_spec, tuple):
                # It's a list of attributes [a1, a2,...]. Satisfied if ANY is present.
                # Logic: P(y) -> (I(a1) v I(a2))
                # Using Max T-conorm for disjunction: I(a1 v a2) = max(I(a1), I(a2))
                composite_conf = max(attribute_confidences.get(a, 0.0) for a in attr_spec)
                best_attr = max(attr_spec, key=lambda a: attribute_confidences.get(a, 0.0))
                violation_magnitude = distance_to_satisfaction_implication(
                    class_posterior, composite_conf
                )
            else:
                # Standard Simple Requirement
                best_attr = attr_spec
                attr_conf = attribute_confidences.get(attr_spec, 0.0)
                violation_magnitude = distance_to_satisfaction_implication(
                    class_posterior, attr_conf
                )

            if violation_magnitude > 0:
                violations.append(ConstraintViolation(
                    constraint_type="necessity",
                    attribute=best_attr,
                    violation_magnitude=violation_magnitude * self.rule_weight,
                    class_name=class_name,
                    attribute_group=tuple(attr_spec) if isinstance(attr_spec, (list, tuple)) else None
                ))
                total_energy += violation_magnitude * self.rule_weight
        
        for attr_spec in forb_groups:
            if isinstance(attr_spec, (list, tuple)):
                # Handle composite exclusion: P(y) -> ¬(a1 v a2)
                # Group confidence: max(a1, a2)
                composite_conf = max(attribute_confidences.get(a, 0.0) for a in attr_spec)
                best_attr = max(attr_spec, key=lambda a: attribute_confidences.get(a, 0.0))
                violation_magnitude = distance_to_satisfaction_exclusion(
                    class_posterior, composite_conf
                )
                if violation_magnitude > 0:
                    violations.append(ConstraintViolation(
                        constraint_type="exclusion",
                        attribute=best_attr,
                        violation_magnitude=violation_magnitude * self.rule_weight,
                        class_name=class_name,
                        attribute_group=tuple(attr_spec)
                    ))
                    total_energy += violation_magnitude * self.rule_weight
            else:
                attr = attr_spec
                attr_conf = attribute_confidences.get(attr, 0.0)
                violation_magnitude = distance_to_satisfaction_exclusion(
                    class_posterior, attr_conf
                )
                if violation_magnitude > 0:
                    violations.append(ConstraintViolation(
                        constraint_type="exclusion",
                        attribute=attr,
                        violation_magnitude=violation_magnitude * self.rule_weight,
                        class_name=class_name,
                        attribute_group=None
                    ))
                    total_energy += violation_magnitude * self.rule_weight
        
        return total_energy, violations
    
    def compute_posteriors_from_graph_scores(
        self,
        graph_scores: GraphScores
    ) -> Dict[str, float]:
        """
        Convert graph scores to posterior probabilities.
        
        Uses a softmax-like normalization over negative energies.
        
        Args:
            graph_scores: Dict of class names to graph compatibility scores
        
        Returns:
            Dict of class names to posterior probabilities
        """
        if not graph_scores:
            return {}

        energies = {}
        for k, v in graph_scores.items():
            try:
                energies[k] = float(v)
            except (TypeError, ValueError):
                energies[k] = float('inf')

        finite_vals = [v for v in energies.values() if np.isfinite(v)]
        if not finite_vals:
            num_classes = len(energies)
            return {k: 1.0 / num_classes for k in energies}

        min_energy = min(finite_vals)
        scores = {}
        for k, v in energies.items():
            if np.isfinite(v):
                scores[k] = float(np.exp(-(v - min_energy)))
            else:
                scores[k] = 0.0

        total = sum(scores.values())
        if total <= 0:
            num_classes = len(scores)
            return {k: 1.0 / num_classes for k in scores}

        return {k: v / total for k, v in scores.items()}
    
    def find_minimum_energy_class(
        self,
        attribute_confidences: AttributeConfidences,
        graph_scores: GraphScores
    ) -> PSLResult:
        """
        Find the optimal class using HL-MRF optimization (Paper Eq. 3-4).
        
        Solves the convex optimization problem:
            minimize: Σ_y J_PSL(y, a, P(y))
            subject to: P(y) ≥ 0 for all y
                       Σ P(y) = 1 (mutually exclusive classes)
        
        This finds the optimal soft truth values P(y) that minimize total
        inconsistency energy across all constraints.
        
        ŷ_psl = argmax P(y) after optimization
        
        Reference: Paper Section 4.2, Equations (3) and (4)
        
        Args:
            attribute_confidences: VLM-derived attribute confidences
            graph_scores: Graph compatibility scores (used for initialization)
        
        Returns:
            PSLResult with best class, total energy, and violations
        """
        class_names = list(self.kg.get_all_class_names())
        n_classes = len(class_names)
        
        if n_classes == 0:
            return PSLResult(
                total_energy=float('inf'),
                class_name="UNCERTAIN",
                violations=[],
                individual_energies={}
            )
        
        def objective(p_values: np.ndarray) -> float:
            """Total HL-MRF energy to minimize: Σ_y J_PSL(y, a, P(y))."""
            total_energy = 0.0
            for i, class_name in enumerate(class_names):
                p_y = float(p_values[i])
                energy, _ = self.compute_class_energy(
                    class_name, p_y, attribute_confidences
                )
                total_energy += energy
            return total_energy
        
        init_posteriors = self.compute_posteriors_from_graph_scores(graph_scores)
        x0 = np.array([init_posteriors.get(cn, 1.0 / n_classes) for cn in class_names])
        x0 = x0 / x0.sum()
        
        constraints = {'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(n_classes)]
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if (not result.success) or (not np.isfinite(result.x).all()):
            optimal_vec = x0
            objective_energy = float(objective(optimal_vec))
        else:
            optimal_vec = result.x
            objective_energy = float(result.fun)

        optimal_p = {class_names[i]: float(optimal_vec[i]) for i in range(n_classes)}
        
        best_class = max(optimal_p, key=optimal_p.get)
        best_p_value = optimal_p[best_class]
        
        best_class_energy, violations = self.compute_class_energy(
            best_class, best_p_value, attribute_confidences
        )
        
        individual_energies = {}
        for class_name in class_names:
            p_y = optimal_p[class_name]
            energy, _ = self.compute_class_energy(
                class_name, p_y, attribute_confidences
            )
            individual_energies[class_name] = energy
        
        return PSLResult(
            total_energy=best_class_energy,
            class_name=best_class,
            violations=violations,
            individual_energies=individual_energies,
            optimized_posteriors=optimal_p,
            objective_energy=objective_energy
        )
    
    def check_consistency(
        self,
        graph_hypothesis: str,
        psl_result: PSLResult
    ) -> bool:
        """
        Check if graph hypothesis matches PSL result.
        
        Returns True if ŷ_graph == ŷ_psl (no label shift).
        """
        return graph_hypothesis == psl_result.class_name
    
    def get_max_violation(
        self,
        violations: List[ConstraintViolation]
    ) -> ConstraintViolation:
        """
        Get the constraint violation with maximum magnitude.
        
        This is the attribute a* most responsible for inconsistency.
        
        Reference: Paper Equation (6)
        """
        if not violations:
            return None
        return max(violations, key=lambda v: v.violation_magnitude)
