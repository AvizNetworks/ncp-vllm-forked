#!/usr/bin/env bash
set -x

#update the model name here "local-folder-name"
MODEL_NAME="NEW_MODEL_NAME"

# Set the model name from the MODEL environment variable
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"${MODEL}"}
if [[ -z "${MODEL}" ]]; then
    echo "ERROR: Missing environment variable MODEL"
    exit 1
fi

# Additional arguments to pass to the API server on startup
additional_args=${EXTRA_ARGS:-""}
if [[ ! -z "${QUANTIZATION}" ]]; then
    if [[ -z "${DTYPE}" ]]; then
        echo "ERROR: Missing environment variable DTYPE when QUANTIZATION is set"
        exit 1
    else
        additional_args="${additional_args} -q ${QUANTIZATION} --dtype ${DTYPE}"
    fi
elif [[ ! -z "${DTYPE}" ]]; then
    additional_args="${additional_args} --dtype ${DTYPE}"
fi
if [[ ! -z "${MAX_MODEL_LEN}" ]]; then
    additional_args="${additional_args} --max-model-len ${MAX_MODEL_LEN}"
fi

test -n "$MODEL_NAME"
MODEL_DIR="$HOME/models/$MODEL_NAME"
test -d "$MODEL_DIR"
# Start the vLLM OpenAI API compatible server
python3 -O -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model=$HOME/models/$MODEL_NAME \
    --served-model-name "${SERVED_MODEL_NAME}" ${additional_args}
