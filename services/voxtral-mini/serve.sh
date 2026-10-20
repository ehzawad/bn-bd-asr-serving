#!/usr/bin/env bash
# vLLM's OpenAI-compatible server for Voxtral Mini bn-BD on a local GPU, with the same
# settings as the Modal inference service. Send it <= 25 s pieces (client.py does the
# chunking): Voxtral's transcription endpoint does not split long audio itself.
#
#   ./serve.sh                        # http://127.0.0.1:8000, CUDA graphs on
#   VLLM_API_KEY=... ./serve.sh       # require "Authorization: Bearer ..." on /v1/*
#   EAGER=1 PORT=9000 ./serve.sh      # faster start, slower decoding
#
# The model repo is private: log in with `hf auth login` or set HF_TOKEN first.
set -euo pipefail

MODEL=${MODEL:-ehzawad/Voxtral-Mini-3B-2507-bn-BD}
REVISION=${REVISION:-d1c7aeb0cd18f8a7a057bf756df5ee2ccb7aebcd}  # the benchmarked snapshot
args=(
  --served-model-name voxtral-mini-bn-bd
  --tokenizer-mode mistral --config-format mistral --load-format mistral
  --max-model-len 2048
  --max-num-seqs "${MAX_NUM_SEQS:-64}"
  --max-num-batched-tokens 16384
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --limit-mm-per-prompt '{"audio": 1}'
  --generation-config vllm
  --seed 0
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
)
if [ -d "$MODEL" ]; then
  : # a local Mistral-format directory: no revision
else
  args+=(--revision "$REVISION" --tokenizer-revision "$REVISION")
fi
if [ "${EAGER:-0}" = "1" ]; then args+=(--enforce-eager); fi

exec vllm serve "$MODEL" "${args[@]}" "$@"
