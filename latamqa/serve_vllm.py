#!/usr/bin/env python3
"""
Self-host a registered model with vLLM's OpenAI-compatible server.

Resolves a model defined in ``latamqa/models/*.yaml`` to its Hugging Face
repository (or a local checkpoint) and launches ``vllm serve`` so it can be
evaluated through the same OpenAI-compatible path the rest of the project uses:

    uv run serve_vllm --model llama-3.1-8b
    # then, in another shell:
    uv run eval_mcq --model openai/llama-3.1-8b \\
        --llm_uri http://localhost:8000/v1 --llm_api_key dummy

vLLM is **not** a project dependency (it is a heavy, GPU-only package). Install
it in the serving environment beforehand, e.g. ``uv pip install vllm`` or
``pip install vllm`` on the GPU host.
"""

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

from structlog import get_logger

from latamqa.model_eval import MODELS_DIR, load_models

logger = get_logger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Optional YAML field that overrides the serving source (e.g. a local checkpoint
# path) when it differs from the human-readable Hugging Face ``Model name``.
SERVE_SOURCE_FIELD = "vLLM model"


def resolve_source(model_id: str | None, model_path: str | None) -> tuple[str, str]:
    """Resolve (serving_source, served_name) from the registry and/or CLI overrides.

    Priority for the serving source: ``--model_path`` > YAML ``vLLM model`` field
    > YAML ``Model name`` (the Hugging Face repository id).
    """
    if not model_id and not model_path:
        raise ValueError("Provide --model (a registered model id) and/or --model_path.")

    served_name = model_id
    source = model_path

    if model_id:
        models = load_models(MODELS_DIR)
        if model_id not in models:
            available = ", ".join(sorted(models)) or "<none>"
            raise ValueError(f"Unknown model id «{model_id}». Available: {available}.")
        config = models[model_id]
        if source is None:
            source = config.get(SERVE_SOURCE_FIELD) or config.get("Model name")
        if not source:
            raise ValueError(
                f"Model «{model_id}» has no '{SERVE_SOURCE_FIELD}' or 'Model name' field to serve. "
                f"Pass --model_path to point at a Hugging Face repo or local checkpoint."
            )

    if served_name is None:
        # Standalone serving (no registry entry): name after the checkpoint.
        served_name = Path(source).name

    return source, served_name


def build_command(args: argparse.Namespace, source: str, served_name: str, passthrough: list[str]) -> list[str]:
    """Build the ``vllm serve`` argument vector."""
    cmd = [
        "vllm",
        "serve",
        source,
        "--served-model-name",
        served_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    cmd += passthrough
    return cmd


def main():
    parser = argparse.ArgumentParser(
        prog="serve_vllm",
        description="Self-host a registered LatamQA model with vLLM's OpenAI-compatible server.",
        epilog=(
            "Unrecognised arguments are forwarded verbatim to `vllm serve` "
            "(e.g. --quantization awq --max-num-seqs 64). "
            "See <https://github.com/Inria-Chile/LatamQA> for additional information."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Registered model id (a YAML stem in latamqa/models/, e.g. 'llama-3.1-8b').",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Override the serving source with a Hugging Face repo id or a local checkpoint path.",
    )
    parser.add_argument(
        "--served_name",
        default=None,
        help="Name vLLM exposes the model under (defaults to the model id). Evaluate with `--model openai/<served_name>`.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve on.")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to shard the model across (>=2 required for ~70B models).",
    )
    parser.add_argument("--dtype", default="auto", help="Weights dtype (auto, bfloat16, float16, ...).")
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum context length (vLLM default if unset).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may allocate.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graph capture (lower memory, slower; useful for debugging).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved `vllm serve` command and exit without launching.",
    )

    args, passthrough = parser.parse_known_args()

    try:
        source, served_name = resolve_source(args.model, args.model_path)
    except ValueError as e:
        logger.fatal(str(e))
        sys.exit(2)

    if args.served_name:
        served_name = args.served_name

    cmd = build_command(args, source, served_name, passthrough)

    logger.info(
        "Serving model with vLLM.",
        source=source,
        served_name=served_name,
        url=f"http://{args.host}:{args.port}/v1",
    )
    logger.info(
        "Evaluate against this server with:",
        command=(f"uv run eval_mcq --model openai/{served_name} --llm_uri http://localhost:{args.port}/v1 --llm_api_key dummy"),
    )

    if args.dry_run:
        print(shlex.join(cmd))
        return

    if shutil.which("vllm") is None:
        logger.fatal(
            "The `vllm` executable was not found on PATH. Install it in the serving "
            "environment first, e.g. `uv pip install vllm` (GPU host required)."
        )
        sys.exit(127)

    # Replace this process with vLLM so signals (Ctrl-C, SLURM term) are handled
    # directly by the server.
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()


# 103793196502094375796_9145abc4d42c23de
