"""
Baseline Pipeline for UXO Recognition.

This module implements the baseline approach using ONLY the VLM for direct classification,
without the neuro-symbolic components (Knowledge Graph or PSL).
"""
import json
import re
from typing import Dict, List, Optional

from core.types import InferenceResult, SystemState
from components.vlm_inspector import create_vlm_inspector, VLMProvider


class BaselinePipeline:
    """
    Baseline pipeline that asks the VLM directly for class and risk.
    """
    
    def __init__(self, vlm_provider: str = "mock", **vlm_kwargs):
        """
        Initialize baseline pipeline.
        
        Args:
            vlm_provider: VLM provider type
            **vlm_kwargs: Provider arguments
        """
        # We use the factory just to get the provider instance
        inspector = create_vlm_inspector(vlm_provider, attributes_list=[], **vlm_kwargs)
        self.provider = inspector.provider
        
        self.classes = [
            "Projectile", "Mortar_Bomb", "Grenade", "Rocket", 
            "Aviation_Bomb", "Mine", "RPG", "Cartridge", "AntiSubmarine_Bomb"
        ]
        
        self.risk_levels = ["Inert", "Low", "Medium", "High", "Critical"]
        
    def _build_prompt(self) -> str:
        classes_str = "\n".join([f"- {c}" for c in self.classes])
        risks_str = "\n".join([f"- {r}" for r in self.risk_levels])
        
        return f"""You are an expert EOD (Explosive Ordnance Disposal) technician.
Analyze the object in this image and classify it into one of the following categories:

CLASSES:
{classes_str}


Respond with a JSON object containing the predicted class.
Format:
{{
  "class": "ClassName",
}}

If you are unsure, choose the most likely class based on visual features.
"""

    def run(self, image_path: str) -> InferenceResult:
        """
        Run baseline inference.
        """
        prompt = self._build_prompt()
        print(f"DEBUG: Baseline Prompt:\n{prompt[:200]}...")
        
        try:
            response = self.provider.query(image_path, prompt)
            print(f"DEBUG: Baseline Response (len={len(response)}):")
            print(response[:300] + "..." if len(response)>300 else response)
            
            # Parse JSON
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
            pred_risk = data.get("risk", "UNKNOWN")
            
            # Normalize class name (handle spaces vs underscores)
            pred_class = pred_class.replace(" ", "_")
            if pred_class not in self.classes and pred_class != "UNKNOWN":
              
                 for c in self.classes:
                     if c.lower() in pred_class.lower() or pred_class.lower() in c.lower():
                         pred_class = c
                         break

            return InferenceResult(
                predicted_class=pred_class,
                confidence=1.0, 
                state=SystemState.NORMAL,
                reasoning_trace=[
                    "Baseline VLM Direct Query",
                    f"Prompt: {prompt[:50]}...",
                    f"Response: {response}"
                ],
                iterations_used=1,
                attribute_confidences={},
                vlm_trace=[{"prompt": prompt, "response": response}]
            )
            
        except Exception as e:
            return InferenceResult(
                predicted_class="ERROR",
                #risk_level="UNKNOWN", Future Research to estimate Risks
                confidence=0.0,
                state=SystemState.UNCERTAINTY,
                reasoning_trace=[f"Error: {str(e)}"],
                iterations_used=0
            )
