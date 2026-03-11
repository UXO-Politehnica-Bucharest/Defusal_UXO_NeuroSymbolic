"""
Knowledge Graph Parser for UXO Recognition.

This module loads and processes the Knowledge Graph JSON,
computing graph-based compatibility scores for class inference.

Reference: Paper Section 4.1 - Ontologically-Structured Knowledge Graph
"""
import json
from typing import Dict, List, Optional, Set, Tuple, Union

from core.types import OrdnanceClass, AttributeConfidences, GraphScores


class KnowledgeGraphParser:
    """
    Parses and processes the Knowledge Graph for symbolic inference.
    
    The KG defines:
    - Required attributes (HAS_PART): Class implies attribute
    - Forbidden attributes (EXCLUDES): Class implies NOT attribute  
    """
    
    def __init__(self, json_path: str):
        """
        Initialize the parser with a Knowledge Graph JSON file.
        
        Args:
            json_path: Path to the knowledge_graph.json file
        """
        self.classes: Dict[str, OrdnanceClass] = {}
        self.all_attributes: List[str] = []
        self.attribute_definitions: Dict[str, str] = {}
        self._load(json_path)
        self.active_attributes = self._compute_active_attributes()
    
    def _load(self, path: str) -> None:
        """Load and parse the Knowledge Graph JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.all_attributes = data.get("all_attributes", [])
        self.attribute_definitions = data.get("attribute_definitions", {})
        self.contradictory_pairs = self._sanitize_contradictory_pairs(
            data.get("contradictory_attribute_pairs", [])
        )
        
        for class_name, class_data in data.get("classes", {}).items():
            
            # Parse required attributes handling composite requirements (lists -> tuples)
            raw_required = class_data.get("required_attributes", [])
            processed_required = set()
            for req in raw_required:
                if isinstance(req, list):
                    processed_required.add(tuple(req))
                else:
                    processed_required.add(req)

            # Parse forbidden attributes (preserve composite groups)
            raw_forbidden = class_data.get("forbidden_attributes", [])
            processed_forbidden = set()
            for forbid in raw_forbidden:
                if isinstance(forbid, list):
                    processed_forbidden.add(tuple(forbid))
                else:
                    processed_forbidden.add(forbid)

            self.classes[class_name] = OrdnanceClass(
                name=class_name,
                required_attributes=processed_required,
                forbidden_attributes=processed_forbidden,
                supporting_attributes=class_data.get("supporting_attributes", [])
            )

    def _sanitize_contradictory_pairs(self, pairs: List) -> List[Tuple[str, str]]:
        """Normalize contradictory attribute pairs to valid (attr1, attr2) tuples."""
        clean_pairs = []
        seen = set()
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            attr1, attr2 = pair
            if attr1 == attr2:
                continue
            key = (attr1, attr2)
            if key in seen:
                continue
            seen.add(key)
            clean_pairs.append((attr1, attr2))
        return clean_pairs

    def _compute_active_attributes(self) -> List[str]:
        """Return ordered list of attributes used by constraints or contradictions."""
        used = set()
        for cls in self.classes.values():
            for attr_spec in cls.required_attributes:
                if isinstance(attr_spec, (list, tuple)):
                    used.update(attr_spec)
                else:
                    used.add(attr_spec)
            for attr_spec in cls.forbidden_attributes:
                if isinstance(attr_spec, (list, tuple)):
                    used.update(attr_spec)
                else:
                    used.add(attr_spec)
            for attr in cls.supporting_attributes:
                used.add(attr)

        for attr1, attr2 in self.contradictory_pairs:
            used.add(attr1)
            used.add(attr2)

        ordered = [a for a in self.all_attributes if a in used]
        missing = sorted(a for a in used if a not in self.all_attributes)
        ordered.extend(missing)
        return ordered

    
    def get_all_class_names(self) -> List[str]:
        """Return list of all class names in the KG."""
        return list(self.classes.keys())
    
    def get_class(self, class_name: str) -> Optional[OrdnanceClass]:
        """Get an OrdnanceClass by name."""
        return self.classes.get(class_name)
    
    def compute_kg_energy(
        self, 
        class_name: str, 
        attribute_confidences: AttributeConfidences
    ) -> float:
        """
        Compute J_KG(y|a) for a specific class using energy-based formulation.
        
        J_KG(y, a) = Σ(1 - I(a_i)) for a_i ∈ N_req(y) 
                   + Σ I(a_j) for a_j ∈ N_forbid(y)
        
        Lower energy = better class candidate.
        This is consistent with J_PSL when P(y) = 1.
        
        Reference: Paper Equation (1)
        
        Args:
            class_name: Name of the ordnance class
            attribute_confidences: Dict mapping attribute names to VLM confidences
        
        Returns:
            Energy value >= 0. Lower is better.
        """
        cls = self.classes.get(class_name)
        if cls is None:
            return float('inf')
        

        
        required_energy = 0.0
        for attr_spec in cls.required_attributes:
            if isinstance(attr_spec, (list, tuple)):
                composite_conf = max(attribute_confidences.get(a, 0.0) for a in attr_spec)
                required_energy += (1.0 - composite_conf)
            else:
                required_energy += (1.0 - attribute_confidences.get(attr_spec, 0.0))

        forbidden_energy = 0.0
        for attr_spec in cls.forbidden_attributes:
            if isinstance(attr_spec, (list, tuple)):
                forbidden_energy += max(attribute_confidences.get(a, 0.0) for a in attr_spec)
            else:
                forbidden_energy += attribute_confidences.get(attr_spec, 0.0)

        return required_energy + forbidden_energy
    
    def compute_all_kg_energies(
        self, 
        attribute_confidences: AttributeConfidences
    ) -> GraphScores:
        """
        Compute J_KG for all classes.
        
        Args:
            attribute_confidences: VLM-derived attribute confidences
        
        Returns:
            Dict mapping class names to energy values
        """
        return {
            class_name: self.compute_kg_energy(class_name, attribute_confidences)
            for class_name in self.classes
        }
    
    def get_initial_hypothesis(
        self, 
        attribute_confidences: AttributeConfidences
    ) -> Tuple[str, float]:
        """
        Get initial graph hypothesis: ŷ_graph = argmin J_KG(y|a).
        
        Reference: Paper Equation (2)
        
        Args:
            attribute_confidences: VLM-derived attribute confidences
        
        Returns:
            Tuple of (best_class_name, min_energy)
        """
        energies = self.compute_all_kg_energies(attribute_confidences)
        best_class = min(energies, key=energies.get)
        return best_class, energies[best_class]
    
    def is_uncertain(
        self, 
        class_name: str, 
        attribute_confidences: AttributeConfidences
    ) -> bool:
        """
        Check if hypothesis indicates uncertainty based on KG energy.
        
        Args:
            class_name: The predicted class name
            attribute_confidences: VLM confidences
        
        Returns:
            True if uncertainty state should be triggered (min J_KG > |N_req|)
        """
        cls = self.classes.get(class_name)
        if cls is None:
            return True

        required_count = len(cls.required_attributes)
        if required_count == 0:
            return False

        total_energy = self.compute_kg_energy(class_name, attribute_confidences)
        return total_energy > float(required_count)
    
    
    def get_required_attributes(self, class_name: str) -> Set[Union[str, Tuple[str, ...]]]:
        """Get required attributes for a class."""
        cls = self.classes.get(class_name)
        return cls.required_attributes if cls else set()
    
    def get_forbidden_attributes(self, class_name: str) -> Set[Union[str, Tuple[str, ...]]]:
        """Get forbidden attributes for a class."""
        cls = self.classes.get(class_name)
        return cls.forbidden_attributes if cls else set()

    def count_edges(self) -> int:
        """Count total attribute edges (flattening composite groups)."""
        count = 0
        for cls in self.classes.values():
            for attr_spec in cls.required_attributes:
                count += len(attr_spec) if isinstance(attr_spec, (list, tuple)) else 1
            for attr_spec in cls.forbidden_attributes:
                count += len(attr_spec) if isinstance(attr_spec, (list, tuple)) else 1
        return count
