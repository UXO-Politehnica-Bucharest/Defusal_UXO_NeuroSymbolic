"""
Baseline Pipeline for UXO Recognition with Attribute Constraints.
"""
import json
import re
from typing import Dict, List, Optional

from core.types import InferenceResult, SystemState
from components.vlm_inspector import create_vlm_inspector, VLMProvider


class BaselineAttributesPipeline:
    """
    Baseline pipeline that asks the VLM directly for class,
    but includes explicit REQUIRED and FORBIDDEN attributes in the prompt.
    """
    
    def __init__(self, kg_path: str = "data/knowledge_graph.json", vlm_provider: str = "mock", **vlm_kwargs):
        """
        Initialize baseline attributes pipeline.
        
        Args:
            kg_path: Path to knowledge graph JSON
            vlm_provider: VLM provider type
            **vlm_kwargs: Provider arguments
        """
        # We use the factory just to get the provider instance
        inspector = create_vlm_inspector(vlm_provider, attributes_list=[], **vlm_kwargs)
        self.provider = inspector.provider
        
        # Load KG
        try:
            with open(kg_path, 'r') as f:
                self.kg = json.load(f)
        except Exception as e:
            print(f"Error loading KG from {kg_path}: {e}")
            self.kg = {"classes": {}}

        self.classes = list(self.kg.get("classes", {}).keys())
        self.risk_levels = ["Inert", "Low", "Medium", "High", "Critical"]
        
    def _format_constraints(self) -> str:
        """
        Format the constraints (required/forbidden) for each class into a string.
        Also includes definitions of the attributes used.
        """
        lines = []
        used_attributes = set()
        
        # Sort classes to ensure deterministic prompt
        for cls_name in sorted(self.classes):
            cls_data = self.kg["classes"][cls_name]
            lines.append(f"- {cls_name}:")
            
            # Required Attributes
            reqs = cls_data.get("required_attributes", [])
            req_strings = []
            for group in reqs:
                if isinstance(group, list):
                    if group:
                        group_str = " OR ".join(group)
                        req_strings.append(f"({group_str})")
                        used_attributes.update(group)
                elif isinstance(group, str):
                    req_strings.append(group)
                    used_attributes.add(group)
            
            if req_strings:
                lines.append(f"  * REQUIRED (Must show): {' AND '.join(req_strings)}")
                
            # Forbidden Attributes
            forb = cls_data.get("forbidden_attributes", [])
            forb_strings = []
            for group in forb:
                if isinstance(group, list):
                     forb_strings.extend(group)
                     used_attributes.update(group)
                elif isinstance(group, str):
                    forb_strings.append(group)
                    used_attributes.add(group)
            
            # Deduplicate and sort
            forb_strings = sorted(list(set(forb_strings)))
            
            if forb_strings:
                 lines.append(f"  * FORBIDDEN (Must NOT show): {', '.join(forb_strings)}")
            
            lines.append("") # Empty line between classes
        
        # Add Unknown class option
        lines.append("- Unknown:")
        lines.append("  * CONSTRAINT: Select this class ONLY if the visible features do NOT match the REQUIRED attributes of any other class.")
        lines.append("")

        # Add definitions relative to used attributes
        lines.append("VISUAL ATTRIBUTE DEFINITIONS:")
        definitions = self.kg.get("attribute_definitions", {})
        for attr in sorted(used_attributes):
            if attr in definitions:
                lines.append(f"- {attr}: {definitions[attr]}")
                 
        return "\n".join(lines)

    def _build_prompt(self) -> str:
        constraints = self._format_constraints()
        
        return f"""You are an expert EOD (Explosive Ordnance Disposal) technician.
Analyze the object in this image and classify it into one of the following categories, adhering strictly to the visual constraints defined below.

CLASSES AND CONSTRAINTS:
{constraints}

Respond with a JSON object containing the predicted class.
Format:
{{
  "class": "ClassName",
}}

If the object fits multiple classes, choose the one with the most matching REQUIRED attributes.
"""

    def run(self, image_path: str) -> InferenceResult:
        """
        Run baseline inference with attributes.
        """
        prompt = self._build_prompt()
        print(f"DEBUG: BaselineAttributes Prompt:\n{prompt[:500]}...")
        
        try:
            response = self.provider.query(image_path, prompt)
            print(f"DEBUG: Response (len={len(response)}):")
            print(response[:300] + "..." if len(response)>300 else response)
            

            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
  
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group()
            
            data = json.loads(cleaned)
            
            pred_class = data.get("class", "UNKNOWN")
            

            pred_class = pred_class.replace(" ", "_")
            
            system_state = SystemState.NORMAL
            confidence = 1.0
            
            if pred_class.upper() == "UNKNOWN":
                 pred_class = "UNCERTAIN" # Map to evaluated class name for consistency if needed, or keep Unknown
                 system_state = SystemState.UNCERTAINTY
                 confidence = 0.0
            
            elif pred_class not in self.classes and pred_class != "UNKNOWN":
                 # Try to find closest match
                 for c in self.classes:
                     if c.lower() in pred_class.lower() or pred_class.lower() in c.lower():
                         pred_class = c
                         break

            return InferenceResult(
                predicted_class=pred_class,
                confidence=confidence, 
                state=system_state,
                reasoning_trace=[
                    "Baseline Attributes VLM Query",
                    f"Prompt: {prompt[:100]}...",
                    f"Response: {response}"
                ],
                iterations_used=1,
                attribute_confidences={},
                vlm_trace=[{"prompt": prompt, "response": response}]
            )
            
        except Exception as e:
            return InferenceResult(
                predicted_class="ERROR",
                confidence=0.0,
                state=SystemState.UNCERTAINTY,
                reasoning_trace=[f"Error: {str(e)}"],
                iterations_used=0
            )
