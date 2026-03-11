# DEFUSAL - Neuro‑Symbolic UXO Framework (Anonymous)

Short guide for running the neuro‑symbolic UXO pipeline and evaluations.

## License
Code is released under Creative Commons Attribution 4.0 International (CC BY 4.0).
Some third‑party models and tools referenced here are licensed separately.
Please review and follow those licenses when using this code or any pre‑trained weights.

## What this repo does
- Extracts visual attributes with a VLM
- Scores classes using a Knowledge Graph (KG) energy
- Validates consistency with PSL (Lukasiewicz logic)
- Runs a feedback loop (if necessary)
- Escalates uncertain cases

## Requirements
- Python 3.10+
- Core deps: `numpy`, `scipy`, `requests`, `Pillow`, `torch`, `transformers`
- Optional:
  - OpenAI API: set `OPENAI_API_KEY`
  - HuggingFace Inference API: set `HF_TOKEN`
  - Local vLLM: run a compatible server (OpenAI‑style endpoint)

## Dataset format
JSON list of samples:
```
[
  {"image_path": "path/to/image.jpg", "class_name": "Mortar_Bomb"},
  ...
]
```


## Run evaluation (neuro‑symbolic)
```
python evaluate.py -d data/uxo_dataset.json -p huggingface --hf-model Qwen/Qwen3-VL-32B-Instruct -o results/run.json
```
Other providers: `mock`, `openai`, `local`, `transformers`, `nscale`.

## Run baseline
```
python evaluate.py -d data/dataset.json -p huggingface --hf-model Qwen/Qwen3-VL-32B-Instruct --baseline -o results/baseline.json
```

## Compare baseline vs neuro‑symbolic
```
python evaluate.py -d data/dataset.json -p huggingface --hf-model Qwen/Qwen3-VL-32B-Instruct --compare -o results/compare.json
```


## Customize the Knowledge Graph 
Edit:
- `data/knowledge_graph.json`
  - `classes`: required/forbidden attributes per class
  - `attribute_definitions`: visual descriptions used directly in the extraction prompt


## Outputs
Evaluation produces a JSON file with:
- `live_metrics` (accuracy, F1, HRR, etc.)
- `samples` (per‑image predictions, traces, PSL/KG scores, reasoning trace)

---
