"""
VLM Inspector - Attribute Extraction from Vision-Language Models.

This module provides a unified interface for extracting visual attributes
from images using various VLM providers (OpenAI, local vLLM, etc.).

The VLM is used ONLY for primitive attribute extraction, NOT for
direct classification. Classification is done by the symbolic reasoning layer.

"""
import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import requests
from requests.exceptions import Timeout, RequestException

try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
except ImportError:
    torch = None
    Image = None
    AutoProcessor = None
    AutoModelForVision2Seq = None

from core.types import AttributeConfidences, QueryType


TIMEOUT_SECONDS = 90
MAX_RETRIES = 10
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_NEW_TOKENS = 8192


def request_with_retry(
    url: str,
    headers: Optional[Dict] = None,
    json_payload: Optional[Dict] = None,
    timeout: int = TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES
) -> Tuple[Dict, bool]:
    """
    Make HTTP POST request with timeout and retry logic.
    
    Args:
        url: Target URL
        headers: Request headers
        json_payload: JSON payload
        timeout: Timeout in seconds (default 90)
        max_retries: Maximum retry attempts (default 10)
        
    Returns:
        Tuple of (response_json, is_timeout)
        If timeout after retries, returns ({}, True)
    """
    last_error = None
    wait_time = 2  # Start with 2 seconds for server errors
    attempt = 0
    
    while attempt < max_retries:
        response = None
        try:
            response = requests.post(
                url,
                headers=headers,
                json=json_payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json(), False
            
        except Timeout as e:
            last_error = e
            attempt += 1
            if attempt < max_retries:
                print(f"  Timeout on attempt {attempt}/{max_retries}, retrying...")
                continue
                
        except RequestException as e:
            last_error = e
            error_str = str(e)
            
            # Handle Rate Limits (429)
            if "429" in error_str or "Too Many Requests" in error_str:
                import time
                wait_time_429 = 30
                print(f"  Rate limited (429) on attempt {attempt+1}/{max_retries}, waiting {wait_time_429} seconds...")
                time.sleep(wait_time_429)
                attempt += 1  # Rate limit counts as an attempt
                continue
            
            # Handle Server Errors (500, 502, 503, 504)
            is_server_error = False
            if response is not None:
                if response.status_code in [500, 502, 503, 504]:
                    is_server_error = True
            
            # Fallback if response object is not available but error message suggests server error
            if not is_server_error and ("500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str):
                is_server_error = True
                
            if is_server_error:
                import time
                attempt += 1
                if attempt < max_retries:
                    print(f"  Server Error ({last_error}) on attempt {attempt}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 60)  # Exponential backoff up to 60s
                    continue
            
            # If not a retryable error, raise immediately
            print(f"  Non-retryable error: {e}")
            raise e
    
    print(f"  TIMEOUT/FAILURE after {max_retries} attempts. Last error: {last_error}")
    return {}, True


class VLMProvider(ABC):
    """Abstract base class for VLM providers."""
    
    @abstractmethod
    def query(
        self, 
        image_path: str, 
        prompt: str, 
        history: Optional[List[Dict]] = None
    ) -> str:
        """Send a query to the VLM and get response."""
        pass


class OpenAIProvider(VLMProvider):
    """OpenAI GPT Vision provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def query(
        self, 
        image_path: str, 
        prompt: str, 
        history: Optional[List[Dict]] = None
    ) -> str:
        """Query OpenAI Vision API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not set")
        
        image_b64 = self._encode_image(image_path)
        
        messages = history or []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0
        }
        
        result, is_timeout = request_with_retry(
            self.base_url, headers=headers, json_payload=payload
        )
        if is_timeout:
            raise TimeoutError("OpenAI API timeout after retries")
        
        return result["choices"][0]["message"]["content"]


class LocalVLLMProvider(VLMProvider):
    """
    Local vLLM provider using OpenAI-compatible API.
    
    Works with any model served via vLLM's OpenAI-compatible endpoint.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        model: Optional[str] = None
    ):
        """
        Initialize local vLLM provider.
        
        Args:
            base_url: vLLM server URL
            model: Model name (optional, uses server default)
        """
        self.base_url = base_url
        self.model = model
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def query(
        self, 
        image_path: str, 
        prompt: str, 
        history: Optional[List[Dict]] = None
    ) -> str:
        """Query local vLLM server."""
        image_b64 = self._encode_image(image_path)
        
        messages = history or []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        })
        
        payload = {
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0
        }
        
        if self.model:
            payload["model"] = self.model
        
        url = f"{self.base_url}/chat/completions"
        result, is_timeout = request_with_retry(url, json_payload=payload)
        if is_timeout:
            raise TimeoutError("Local vLLM timeout after retries")
        
        return result["choices"][0]["message"]["content"]


class NScaleProvider(VLMProvider):
    """
    NScale provider.
    
    Uses the OpenAI-compatible API endpoint from NScale.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = ""
    ):
        """
        Initialize NScale provider.
        
        Args:
            api_key: NScale service token (defaults to NSCALE_SERVICE_TOKEN env var)
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get("NSCALE_SERVICE_TOKEN")
        self.model = model
        self.base_url = "https://inference.api.nscale.com/v1"
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def query(
        self, 
        image_path: str, 
        prompt: str, 
        history: Optional[List[Dict]] = None
    ) -> str:
        """Query NScale API."""
        if not self.api_key:
            raise ValueError("NScale API key not set (NSCALE_SERVICE_TOKEN)")
        
        image_b64 = self._encode_image(image_path)
        
        messages = history or []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0
        }
        
        url = f"{self.base_url}/chat/completions"
        result, is_timeout = request_with_retry(
            url, headers=headers, json_payload=payload
        )
        if is_timeout:
            raise TimeoutError("NScale API timeout after retries")
        
        return result["choices"][0]["message"]["content"]


class HuggingFaceProvider(VLMProvider):
    """
    HuggingFace Inference API provider.
    
    Uses the HuggingFace router API for models like Qwen3-VL-8B-Instruct.
    OpenAI-compatible endpoint.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen3-VL-8B-Instruct"
    ):
        """
        Initialize HuggingFace provider.
        
        Args:
            api_key: HuggingFace token (defaults to HF_TOKEN env var)
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        self.model = model
        self.base_url = "https://router.huggingface.co/v1"
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def query(
        self, 
        image_path: str, 
        prompt: str, 
        history: Optional[List[Dict]] = None
    ) -> str:
        """Query HuggingFace Inference API."""
        if not self.api_key:
            raise ValueError("HuggingFace API key not set (HF_TOKEN)")
        
        image_b64 = self._encode_image(image_path)
        print(f"Using {self.model}...")
        messages = history or []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0
        }
        
        url = f"{self.base_url}/chat/completions"
        result, is_timeout = request_with_retry(
            url, headers=headers, json_payload=payload
        )
        if is_timeout:
            raise TimeoutError("HuggingFace API timeout after retries")
        
        return result["choices"][0]["message"]["content"]


class MockVLMProvider(VLMProvider):
    """
    Mock VLM provider for testing without actual API calls.
    
    Returns randomized but plausible attribute confidences.
    """
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """Initialize mock provider with optional random seed."""
        import random
        if seed is not None:
            random.seed(seed)
        self._random = random
    
    def query(
        self, 
        image_path: str, 
        prompt: str, 
        history: Optional[List[Dict]] = None
    ) -> str:
        """Return mock JSON response."""
        mock_attributes = {
            "elongated_cylindrical": self._random.random(),
            "spherical_round": self._random.random(),
            "teardrop_shaped_front": self._random.random(),
            "tail_fins": self._random.random(),
            "driving_band_visible": self._random.random(),
            "fits_in_hand": self._random.random(),
            "rust_visible": self._random.random(),
        }
        # In a real scenario, we might want to vary this based on the image path or prompt
        # for more realistic mock testing.
        return json.dumps(mock_attributes)


class TransformersProvider(VLMProvider):
    """
    Provider for HuggingFace Transformers models.
    
    Loads models directly using AutoModelForVision2Seq and AutoProcessor.
    Supports quantization via bitsandbytes if available (and if loaded with load_in_4bit=True).
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize Transformers provider.
        
        Args:
            model_path: HuggingFace model ID or local path
            device: Device to load model on
            **kwargs: Additional args passed to from_pretrained (e.g. load_in_4bit)
        """
        self.device = device
        self.model_path = model_path
        
        print(f"Loading model {model_path} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path, 
            device_map=device if device != "cpu" else None,
            **kwargs
        )
        if device == "cpu":
             self.model.to("cpu")

    def query(
        self, 
        image_path: str, 
        prompt: str, 
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        Query the HuggingFace model.
        """
        print(f"\n{'='*60}")
        print(f"[DEBUG VLM] Image path: {image_path}")
        print(f"[DEBUG VLM] Image exists: {os.path.exists(image_path)}")
        
        image = Image.open(image_path).convert("RGB")
        print(f"[DEBUG VLM] Image size: {image.size}")
        print(f"[DEBUG VLM] Image mode: {image.mode}")
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        try:
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            print(f"[DEBUG VLM] Using chat template")
        except (ValueError, AttributeError) as e:
            print(f"[DEBUG VLM] Chat template failed: {e}")
            text_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            print(f"[DEBUG VLM] Using fallback prompt format")
        
        print(f"[DEBUG VLM] Final prompt (first 500 chars):\n{text_prompt[:500]}...")
            
        inputs = self.processor(
            text=[text_prompt], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        )
        
        # Check model limits
        if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'model_max_length'):
            max_len = self.processor.tokenizer.model_max_length
            current_len = inputs['input_ids'].shape[1]

            if current_len + 1000 > max_len: # Warning if close to limit
                 print(f"[DEBUG VLM] WARNING: Input length ({current_len}) is close to model limit ({max_len})!")
        
        if 'pixel_values' in inputs:
            print(f"[DEBUG VLM] pixel_values shape: {inputs['pixel_values'].shape}")
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=False
            )
        
        print(f"[DEBUG VLM] Output shape: {output_ids.shape}")
       
        input_length = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_length:]
                
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        print(f"[DEBUG VLM] Raw output text:\n{output_text}")
        print(f"{'='*60}\n")
        
        return output_text


class VLMInspector:
    """
    Main VLM Inspector class for attribute extraction.
    
    Provides methods for:
    - Extracting all attributes from an image
    - Re-querying specific attributes during feedback loop
    - Parsing VLM responses into confidence dictionaries
    """
    
    def __init__(
        self, 
        provider: VLMProvider,
        attributes_list: List[str],
        attribute_definitions: Optional[Dict[str, str]] = None,
        contradictory_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Initialize the VLM Inspector.
        
        Args:
            provider: VLM provider instance
            attributes_list: List of all attribute names to extract
            attribute_definitions: Dictionary of attribute visual descriptions
        """
        self.provider = provider
        self.attributes = attributes_list
        self.history = []
        self.ATTRIBUTE_DEFINITIONS = attribute_definitions or {}
        self.contradictory_pairs = [
            pair for pair in (contradictory_pairs or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        self.attribute_aliases = {
            "egg_shaped_oval": "rounded_shape_oval",
            "pear_shaped_body": "teardrop_shaped_front",
            "cast_iron_serrated": "segmented_body_pattern",
            "serrated_body_pattern": "segmented_body_pattern",
            "segmented_body_grid": "segmented_body_pattern",
            "pressure_plate_top": "pressure_plate",
            "cylindric_simple": "cylindrical_simple",
            "long_tubular_body": "slender_tubular_body",
            "fin_stabilized_tubular_tail": "tail_fins",
            "tail_boom_assembly": "perforated_tubular_stalk",
            "separated_components": "separated_components_joined",
            "driving_band_visible": "copper_band_at_base",
            "multiple_driving_bands_at_base": "copper_band_at_base",
            "driving_band": "copper_band_at_base",
            "rotating_band_copper": "copper_band_at_base",
            "separed_components_joined": "separated_components_joined",
            "pull_ring_visible": "pull_ring",
            "continuous_cylindrical_body": "uniform_diameter_profile",
            "pointed_nose": "conical_pointed",
            "ogive_flat": "ogive_blunt",
            "cylindric_bi_ogival": "ogive_conic_elongated",
            "double_conic_short": "ogive_conic_short",
            "nozzle_exhaust": "rear_termination_open",
            "spherical_round": "hemispherical_half",
            "vehicle_scale": "requires_equipment",
            "primer_cap": "base_center_dented",
            "primer_dent": "base_center_dented",
            "case_rim": "rim_flange_base",
            "case_rim_visible": "rim_flange_base",
            "rim_flange": "rim_flange_base",
            "protruding_orthogonal_planes": "tail_fins",
        }
    
    def clear_history(self):
        """Clear the interaction history."""
        self.history = []
    
    def _build_extraction_prompt(self) -> str:
        """Build the prompt for initial attribute extraction using visual primitives only."""
        
        # Build prompt dynamically from the dictionary
        attr_descriptions = []
        for attr in self.attributes:
            desc = self.ATTRIBUTE_DEFINITIONS.get(attr, "Look for this visual feature.")
            attr_descriptions.append(f'- "{attr}": {desc}')

        formatted_descriptions = "\n".join(attr_descriptions)
        template_lines = [f'  "{attr}": 0.0' for attr in self.attributes]
        template_json = "{\n" + ",\n".join(template_lines) + "\n}"
        pairs_text = ""
        if self.contradictory_pairs:
            pair_lines = [f'- {a} vs {b}' for a, b in self.contradictory_pairs]
            pairs_text = (
                "\nCRITICAL MUTUAL EXCLUSIVITY RULES:\n"
                "If you see strong evidence for the first attribute, you MUST score the second as 0.0.\n"
                "Physically impossible combinations:\n"
                + "\n".join(pair_lines)
            )

        guidelines = (
            "\nSTRICT VISUAL RULES:\n"
            "- Default is 0.0 unless there is visual evidence.\n"
            "- Use 0.3-0.5 for faint/partial evidence.\n"
            "- Most attributes should be 0.0; only mark non-zero if you can point to a visible cue.\n"
            "- Do not infer class or function.\n"
        )

        return f"""Analyze the object in this image. Score each attribute 0.0-1.0
based ONLY on visual evidence.
SCORING GUIDELINES:
- 0.0: not visible
- 0.3-0.5: partially visible / uncertain
- 0.6-0.8: clearly visible
- 0.9-1.0: very certain
{pairs_text}

VISUAL ATTRIBUTE DEFINITIONS (use JSON keys with underscores):
{formatted_descriptions}

OUTPUT FORMAT:
- Return a JSON object with ALL keys listed below.
- Start from this template and only change values you see
evidence for:
{template_json}
Missing keys will be treated as 0.0, but do not omit keys.
Example format: {{"tail_fins": 0.7, 
"teardrop_shaped_front": 0.6}}
{guidelines}"""


        
    def _find_matching_attribute(self, vlm_attr: str) -> Optional[str]:
        """
        Find matching attribute using CONTAINS logic for non-standard names.
        
        If VLM returns 'cylindrical' instead of 'elongated_cylindrical',
        finds the correct attribute by substring matching.
        
        Args:
            vlm_attr: Attribute name from VLM (may be non-standard)
        
        Returns:
            Matched attribute name or None if no match
        """
        normalized = vlm_attr.lower().replace(" ", "_").replace("-", "_")
        alias = self.attribute_aliases.get(normalized)
        if alias and alias in self.attributes:
            return alias

        vlm_lower = normalized.replace("_", " ")
        
        for attr in self.attributes:
            attr_lower = attr.lower().replace("_", " ")
            if vlm_lower in attr_lower or attr_lower in vlm_lower:
                return attr

        tokens = set(normalized.split("_"))
        best_attr = None
        best_overlap = 0
        for attr in self.attributes:
            attr_tokens = set(attr.lower().split("_"))
            overlap = len(tokens & attr_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_attr = attr

        if best_attr and (best_overlap >= 2 or (len(tokens) == 1 and best_overlap == 1)):
            return best_attr

        return None
    
    def _parse_json_response(self, response: str) -> AttributeConfidences:
        """
        Parse VLM response into attribute confidences.
        
        Handles various response formats including markdown code blocks.
        Also handles truncated JSON by repairing incomplete responses.
        """
        print(f"[DEBUG PARSE] Attempting to parse response of length {len(response)}")
        
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        print(f"[DEBUG PARSE] After markdown cleanup: {cleaned[:200]}...")

        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        
        if not json_match and '{' in cleaned:
             print(f"[DEBUG PARSE] Strict JSON regex failed, checking for truncated JSON...")
             start_idx = cleaned.find('{')
             candidate = cleaned[start_idx:]
             
             if not candidate.rstrip().endswith('}'):
                 print(f"[DEBUG PARSE] Detected missing closing brace, attempting repair...")
                 last_comma = candidate.rfind(',')
                 if last_comma > 0:
                     repaired = candidate[:last_comma] + '}'
                     print(f"[DEBUG PARSE] Proposed repair (cut at last comma): {repaired[-50:]}")
                     try:
                         json.loads(repaired) # Check validity
                         json_match = re.match(r'\{.*\}', repaired, re.DOTALL) # Mock match
                         json_str = repaired
                     except:
                         repaired_simple = candidate + '}'
                         print(f"[DEBUG PARSE] Repair 1 failed, trying simple append: {repaired_simple[-50:]}")
                         json_str = repaired_simple
                 else:
                     json_str = candidate + '}'
             else:
                 json_str = candidate
        elif json_match:
            json_str = json_match.group()
        else:
            json_str = None

        if json_str:
            print(f"[DEBUG PARSE] Processing JSON candidate of length {len(json_str)}")
            try:
                parsed = json.loads(json_str)
                print(f"[DEBUG PARSE] Successfully parsed JSON with {len(parsed)} keys")
                result = {}
                unmatched = []
                skipped_headers = []
                for key, value in parsed.items():
                    if isinstance(value, bool):
                        value = 1.0 if value else 0.0
                    elif isinstance(value, str):
                        value_str = value.strip()
                        is_percent = value_str.endswith('%')
                        if is_percent:
                            value_str = value_str[:-1]
                        try:
                            value = float(value_str)
                            if is_percent:
                                value = value / 100.0
                        except ValueError:
                            skipped_headers.append(key)
                            continue
                    elif not isinstance(value, (int, float)):
                        skipped_headers.append(key)
                        continue
                    
                    attr_name = key.replace(' ', '_').lower()
                    alias = self.attribute_aliases.get(attr_name, attr_name)
                    if alias in self.attributes:
                        existing = result.get(alias, 0.0)
                        result[alias] = float(max(existing, min(max(value, 0.0), 1.0)))
                    else:
                        matched = self._find_matching_attribute(attr_name)
                        if matched:
                            existing = result.get(matched, 0.0)
                            result[matched] = float(max(existing, min(max(value, 0.0), 1.0)))
                        else:
                            unmatched.append(attr_name)
                if skipped_headers:
                    print(f"[DEBUG PARSE] Skipped category headers: {skipped_headers}")
                if unmatched:
                    print(f"[DEBUG PARSE] Unmatched attributes: {unmatched[:5]}")
                print(f"[DEBUG PARSE] Total matched attributes: {len(result)}")
                return result
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[DEBUG PARSE] JSON parsing failed: {e}")
                print(f"[DEBUG PARSE] Candidate was: {json_str[:200]}...{json_str[-200:] if len(json_str)>200 else ''}")
        else:
            print(f"[DEBUG PARSE] No JSON object found in response")
            print(f"[DEBUG PARSE] Response was: {response[:500]}...")
        
        return {attr: 0.0 for attr in self.attributes}
    
    def extract_attributes(
        self, 
        image_path: str
    ) -> AttributeConfidences:
        """
        Extract all attributes from an image with retry on failure.
        
        If VLM returns zero attributes, retries once.
        If still fails, marks extraction_failed=True for UNCERTAIN state.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dict mapping attribute names to confidence scores
        """
        self.extraction_failed = False
        self.failure_reason = None
        
        prompt = self._build_extraction_prompt()
        print(f"\n[DEBUG INSPECTOR] Extraction prompt:\n{prompt[:300]}...")
        
        for attempt in range(2):
            response = self.provider.query(image_path, prompt)
            print(f"\n[DEBUG INSPECTOR] VLM Response received (attempt {attempt+1}), length: {len(response)}")
            print(f"[DEBUG INSPECTOR] Response preview: {response[:500]}")
            
            confidences = self._parse_json_response(response)
            
            detected_count = sum(1 for v in confidences.values() if v > 0.0)
            print(f"[DEBUG INSPECTOR] Parsed {detected_count} non-zero attributes")
            
            if detected_count:
                self.history.append({
                    "step": "extraction",
                    "attempt": attempt + 1,
                    "prompt": prompt,
                    "response": response,
                    "parsed": confidences
                })
                for attr in self.attributes:
                    if attr not in confidences:
                        confidences[attr] = 0.0
                return confidences
            
            if attempt == 0:
                print(f"[DEBUG INSPECTOR] WARNING: Only {detected_count} attributes - retrying...")
        
        print(f"[DEBUG INSPECTOR] FAILURE: No attributes after 2 attempts")
        self.extraction_failed = True
        self.failure_reason = f"VLM returned {detected_count} attributes after 2 attempts"
        
        self.history.append({
            "step": "extraction_failed",
            "reason": self.failure_reason,
            "response": response
        })
        
        for attr in self.attributes:
            if attr not in confidences:
                confidences[attr] = 0.0
        
        return confidences
    
    def query_specific_attribute(
        self,
        image_path: str,
        attribute: str,
        query_type: QueryType,
        previous_confidence: Optional[float] = None,
        override_prompt: Optional[str] = None
    ) -> float:
        """
        Re-query the VLM for a specific attribute.
        
        Used during the feedback loop to get updated confidence
        for a single attribute that caused a constraint violation.
        
        Args:
            image_path: Path to the image file
            attribute: Attribute name to query
            query_type: COMPLETENESS or CONTRADICTION
            previous_confidence: Previous confidence value (for context)
        
        Returns:
            Updated confidence score for the attribute
        """
        attr_readable = attribute.replace('_', ' ')
        
        if override_prompt:
            prompt = override_prompt.rstrip()
        elif query_type == QueryType.COMPLETENESS:
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
        else:
            prev_conf = previous_confidence if previous_confidence is not None else 0.0
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
        
        response = self.provider.query(image_path, prompt)
        
        result = previous_confidence if previous_confidence is not None else 0.5
        
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response)
        found = False
        if conf_match:
            try:
                result = float(conf_match.group(1))
                found = True
            except ValueError:
                pass
        
        if not found:
            num_match = re.search(r'(0?\.\d+|1\.0|0|1)', response)
            if num_match:
                try:
                    result = float(num_match.group(1))
                except ValueError:
                    pass
        
        self.history.append({
            "step": "feedback_query",
            "attribute": attribute,
            "query_type": query_type.value,
            "prompt": prompt,
            "response": response,
            "parsed_confidence": result
        })
        
        return result

    def query_batch_attributes(
        self,
        image_path: str,
        attributes: List[str],
        query_types: Dict[str, QueryType],
        previous_confidences: Dict[str, float],
        context_hint: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Query multiple attributes in a single VLM call for efficiency.
        
        Args:
            image_path: Path to the image file
            attributes: List of attribute names to re-evaluate
            query_types: Dict mapping attribute to its query type
            previous_confidences: Dict of previous confidence values
            context_hint: Optional context to focus attention on a region/part
        
        Returns:
            Dict mapping attribute names to updated confidence scores
        """
        if not attributes:
            return {}
        
        # Build batch prompt
        attr_lines = []
        for attr in attributes:
            attr_readable = attr.replace('_', ' ')
            prev_conf = previous_confidences.get(attr, 0.0)
            qtype = query_types.get(attr, QueryType.COMPLETENESS)
            definition = self.ATTRIBUTE_DEFINITIONS.get(attr, "")
            definition_text = f"{definition} " if definition else ""
            
            if qtype == QueryType.COMPLETENESS:
                attr_lines.append(
                    f'- "{attr_readable}": {definition_text}Look carefully, is there any evidence? '
                    f'(previous: {prev_conf:.2f})'
                )
            else:
                attr_lines.append(
                    f'- "{attr_readable}": {definition_text}Verify detection, was it real? '
                    f'(previous: {prev_conf:.2f})'
                )
        
        attrs_text = '\n'.join(attr_lines)
        
        context_block = ""
        if context_hint:
            context_block = f"CONTEXT:\n{context_hint}\n\n"

        prompt = f"""Re-evaluate MULTIPLE features on this object.

{context_block}For each feature listed below, provide your UPDATED confidence (0.0-1.0).
Only score a feature above 0.0 if you can see clear visual evidence.
If the relevant region is not visible, keep the score at 0.0.
Look very carefully at the image. Features may be partially obscured.

FEATURES TO RE-EVALUATE:
{attrs_text}

Respond with a JSON object containing ALL features and their confidence:
{{
  "attribute_name": confidence_value,
  ...
}}

Example format:
{{
  "flat disc circular": 1.0,
  "pressure plate top": 0.0
}}"""

        response = self.provider.query(image_path, prompt)
        
  
        results = {}
        try:
            # Clean markdown if present
            cleaned = response.strip()
            if cleaned.startswith('```'):
                lines = cleaned.split('\n')
                cleaned = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            import json
            parsed = json.loads(cleaned)
            
            for attr in attributes:
                attr_readable = attr.replace('_', ' ')
             
                if attr in parsed:
                    results[attr] = float(parsed[attr])
                elif attr_readable in parsed:
                    results[attr] = float(parsed[attr_readable])
                else:
                    results[attr] = previous_confidences.get(attr, 0.0)
        except (json.JSONDecodeError, ValueError):
          
            for attr in attributes:
                results[attr] = previous_confidences.get(attr, 0.0)
        
        # Log the batch query
        self.history.append({
            "step": "batch_feedback_query",
            "attributes": attributes,
            "prompt": prompt,
            "response": response,
            "parsed_results": results
        })
        
        return results

def create_vlm_inspector(
    provider_type: str = "mock",
    attributes_list: Optional[List[str]] = None,
    attribute_definitions: Optional[Dict[str, str]] = None,
    contradictory_pairs: Optional[List[Tuple[str, str]]] = None,
    **kwargs
) -> VLMInspector:
    """
    Factory function to create a VLM Inspector with specified provider.
    
    Args:
        provider_type: One of "openai", "local", "mock", "transformers", "nscale", "huggingface"
        attributes_list: List of attributes to extract
        attribute_definitions: Dictionary of attribute visual descriptions
        **kwargs: Additional arguments for the provider
    
    Returns:
        Configured VLMInspector instance
    """
    if attributes_list is None or attribute_definitions is None or contradictory_pairs is None:
        from components.knowledge_graph import KnowledgeGraphParser
        kg = KnowledgeGraphParser("data/knowledge_graph.json")
        if attributes_list is None:
            source_attrs = kg.active_attributes if hasattr(kg, "active_attributes") else kg.all_attributes
            attributes_list = [a for a in source_attrs if not a.startswith("_UNUSED_")]
        if attribute_definitions is None:
            attribute_definitions = kg.attribute_definitions
        if contradictory_pairs is None:
            contradictory_pairs = getattr(kg, "contradictory_pairs", [])
    
    if provider_type == "openai":
        provider = OpenAIProvider(**kwargs)
    elif provider_type == "local":
        provider = LocalVLLMProvider(**kwargs)
    elif provider_type == "mock":
        provider = MockVLMProvider(**kwargs)
    elif provider_type == "transformers":
        if "model_path" not in kwargs:
             raise ValueError("model_path argument is required for transformers provider")
        provider = TransformersProvider(**kwargs)
    elif provider_type == "nscale":
        nscale_kwargs = {k: v for k, v in kwargs.items() if k not in ["model_path"]}
        provider = NScaleProvider(**nscale_kwargs)
    elif provider_type == "huggingface":
        hf_kwargs = {k: v for k, v in kwargs.items() if k not in ["model_path"]}
        provider = HuggingFaceProvider(**hf_kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return VLMInspector(
        provider, 
        attributes_list, 
        attribute_definitions,
        contradictory_pairs=contradictory_pairs
    )
