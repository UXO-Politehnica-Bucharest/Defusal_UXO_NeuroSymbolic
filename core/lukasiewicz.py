"""
Lukasiewicz Fuzzy Logic Operators for Probabilistic Soft Logic.

This module implements the Lukasiewicz t-norm operators used in the
PSL consistency validation. These operators allow reasoning over
continuous truth values in [0, 1] instead of binary logic.

"""


def implication(a: float, b: float) -> float:
    """
    Lukasiewicz implication: A → B = min(1, 1 - A + B).
    
    Returns the continuous truth value of the implication.
    
    Args:
        a: Truth value of antecedent, in [0, 1]
        b: Truth value of consequent, in [0, 1]
    
    Returns:
        Truth value of implication, in [0, 1]
    
    Example:
        >>> implication(1.0, 0.0)
        0.0
    """
    return min(1.0, 1.0 - a + b)


def negation(a: float) -> float:
    """
    Lukasiewicz negation: ¬A = 1 - A.
    
    Args:
        a: Truth value to negate, in [0, 1]
    
    Returns:
        Negated truth value, in [0, 1]
    """
    return 1.0 - a


def conjunction(a: float, b: float) -> float:
    """
    Lukasiewicz conjunction (AND): A ∧ B = max(0, A + B - 1).
    
    Args:
        a: First operand, in [0, 1]
        b: Second operand, in [0, 1]
    
    Returns:
        Conjunction result, in [0, 1]
    """
    return max(0.0, a + b - 1.0)


def disjunction(a: float, b: float) -> float:
    """
    Lukasiewicz disjunction (OR): A ∨ B = min(1, A + B).
    
    Args:
        a: First operand, in [0, 1]
        b: Second operand, in [0, 1]
    
    Returns:
        Disjunction result, in [0, 1]
    """
    return min(1.0, a + b)


def distance_to_satisfaction_implication(a: float, b: float) -> float:
    """
    Distance to satisfaction for implication A → B.
    
    d_sat(A → B) = max(0, A - B)
    
    This is the "cost" of violating the implication constraint.
    Used for necessity rules: Class → Attribute (if class is present,
    attribute should be present).
    
    Args:
        a: Truth value of antecedent (e.g., class posterior P(y))
        b: Truth value of consequent (e.g., attribute confidence I(a))
    
    Returns:
        Violation magnitude, in [0, 1]. 0 means satisfied.
    
    Reference: Paper Equation (4), first term.
    """
    return max(0.0, a - b)


def distance_to_satisfaction_exclusion(a: float, b: float) -> float:
    """
    Distance to satisfaction for exclusion constraint A → ¬B.
    
    d_sat(A → ¬B) = max(0, A + B - 1)
    
    This penalizes configurations where both the class and a
    forbidden attribute are simultaneously confident.
    
    Args:
        a: Truth value of antecedent (e.g., class posterior P(y))
        b: Truth value of the forbidden attribute confidence I(a)
    
    Returns:
        Violation magnitude, in [0, 1]. 0 means satisfied.
    
    Reference: Paper Equation (4), second term.
    """
    return max(0.0, a + b - 1.0)


def compute_implication_energy(
    class_posterior: float,
    required_attributes: list,
    attribute_confidences: dict
) -> float:
    """
    Compute total energy for necessity constraints.
    
    Σ max(0, P(y) - I(a_i)) for all a_i in N_req(y)
    
    Args:
        class_posterior: Posterior probability of the class P(y)
        required_attributes: List of required attribute names
        attribute_confidences: Dict mapping attribute names to confidences
    
    Returns:
        Total necessity energy
    """
    total = 0.0
    for attr in required_attributes:
        attr_conf = attribute_confidences.get(attr, 0.0)
        total += distance_to_satisfaction_implication(class_posterior, attr_conf)
    return total


def compute_exclusion_energy(
    class_posterior: float,
    forbidden_attributes: list,
    attribute_confidences: dict
) -> float:
    """
    Compute total energy for exclusion constraints.
    
    Σ max(0, P(y) + I(a_j) - 1) for all a_j in N_forbid(y)
    
    Args:
        class_posterior: Posterior probability of the class P(y)
        forbidden_attributes: List of forbidden attribute names
        attribute_confidences: Dict mapping attribute names to confidences
    
    Returns:
        Total exclusion energy
    """
    total = 0.0
    for attr in forbidden_attributes:
        attr_conf = attribute_confidences.get(attr, 0.0)
        total += distance_to_satisfaction_exclusion(class_posterior, attr_conf)
    return total
