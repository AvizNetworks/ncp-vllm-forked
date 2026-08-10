#!/usr/bin/env bash
set -euo pipefail

#update the model name here "local-folder-name"
MODEL_NAME="${MODEL_NAME:-NEW_MODEL_NAME}"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No nvidia-smi found. Ensure NVIDIA drivers are installed."
    exit 1
fi

# Detect number of GPUs if not provided
if [[ -z "${NUM_GPU:-}" ]]; then
    NUM_GPU="$(nvidia-smi -L | wc -l)"
    export NUM_GPU
fi
echo "NUM_GPU=${NUM_GPU}"

if ! [[ "${NUM_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: No usable NVIDIA GPUs detected."
    exit 1
fi

# Set the model name from the MODEL environment variable
if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: Missing environment variable MODEL"
    exit 1
fi
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"

# Additional arguments to pass to the API server on startup
additional_args=()
if [[ -n "${QUANTIZATION:-}" ]]; then
    if [[ -z "${DTYPE:-}" ]]; then
        echo "ERROR: Missing environment variable DTYPE when QUANTIZATION is set"
        exit 1
    else
        additional_args+=(--quantization "${QUANTIZATION}" --dtype "${DTYPE}")
    fi
elif [[ -n "${DTYPE:-}" ]]; then
    additional_args+=(--dtype "${DTYPE}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    additional_args+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    additional_args+=(--max-model-len "${MAX_MODEL_LEN}")
fi

# --worker-use-ray was removed after v0.13. Use the current backend option only
# when explicitly requested; vLLM defaults to multiprocessing on one host.
if [[ -n "${DISTRIBUTED_EXECUTOR_BACKEND:-}" ]]; then
    additional_args+=(--distributed-executor-backend \
        "${DISTRIBUTED_EXECUTOR_BACKEND}")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # EXTRA_ARGS is retained for compatibility with the existing NCP interface.
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_ARGS})
    additional_args+=("${extra_args[@]}")
fi

MODEL_DIR="${MODEL_DIR:-${HOME}/models/${MODEL_NAME}}"
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    exit 1
fi

PORT="${PORT:-8000}"

# Start the vLLM OpenAI API compatible server
exec python3 -O -u -m vllm.entrypoints.openai.api_server \
    --tensor-parallel-size "${NUM_GPU}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    "${additional_args[@]}"
