"""
!!!!!!!!!
VLLLM SERVER IS AVAIABLE ONLY ON LINUX / For Windows, you have to use WSL - not recommended
!!!!!!!!!
vLLM Server Launcher for Local Vision-Language Models.

This script launches vLLM with an OpenAI-compatible API endpoint
for serving vision-language models locally.


Usage:
    python local_models/vllm_server.py --model llava-hf/llava-1.5-7b-hf --port 8000
"""
import argparse
import logging
import os
import sys
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("vLLM_Launcher")



def check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_vllm_command(
    model: str,
    port: int,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    extra_args: list = None
) -> list:
    """
    Build the vLLM server launch command.
    
    Args:
        model: HuggingFace model path
        port: Port to serve on
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum context length
        extra_args: Additional vLLM arguments
    
    Returns:
        Command as list of strings
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--dtype", "float16",
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    return cmd


def run_server(args, extra_args: list = None):
    """
    Launch the vLLM server.
    
    Args:
        args: Parsed command line arguments
        extra_args: Additional arguments to pass to vLLM
    """
    model = args.model 
    
    if not check_gpu_available():
        logger.warning("No GPU detected. vLLM requires CUDA GPU.")
        logger.warning("For CPU-only testing, use the MockVLMProvider instead.")
        return
    
    cmd = build_vllm_command(
        model=model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        extra_args=extra_args
    )

    logger.info("  Neuro-Symbolic UXO Framework - VLM Server")
    logger.info(f"Model: {model}")
    logger.info(f"API Endpoint: http://localhost:{args.port}/v1")
    logger.info(f"GPU Memory: {args.gpu_memory * 100:.0f}%")
    logger.info(f"Max Context: {args.max_model_len}")
    logger.info("Starting server...")
    
    try:
        os.execv(sys.executable, cmd)
    except Exception as e:
        logger.error(f"Failed to launch vLLM server: {e}")
        logger.error("Make sure vLLM is installed: pip install vllm")
        sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch vLLM server for VLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    # Start with specific model
    python local_models/vllm_server.py --model qwen-vl --port 8001

    # Start with full HF model path
    python local_models/vllm_server.py --model {model_name}


        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Model alias or HuggingFace model path"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )

    args, extra_args = parser.parse_known_args()

    
    run_server(args, extra_args)
