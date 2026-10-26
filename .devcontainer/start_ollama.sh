#!/usr/bin/env bash
# Starts the Ollama server in the background (devcontainer postStartCommand).
# Safe to run again by hand: does nothing if the server already answers.
# Log: /tmp/ollama.log. Models: $OLLAMA_MODELS (ml-study-cache volume).
set -euo pipefail

log_file=/tmp/ollama.log

is_up() {
  ollama list >/dev/null 2>&1
}

if ! is_up; then
  mkdir -p "${OLLAMA_MODELS:-$HOME/.ollama/models}"
  # setsid + nohup: keep the server alive after this script exits.
  setsid nohup ollama serve >"$log_file" 2>&1 </dev/null &
  for _ in $(seq 1 30); do
    is_up && break
    sleep 0.5
  done
fi

if ! is_up; then
  echo "ollama: server did not start, see $log_file" >&2
  exit 0
fi

echo "ollama: running on ${OLLAMA_HOST:-127.0.0.1:11434}"
models="$(ollama list)"
if ! grep -q '^mistral' <<<"$models"; then
  echo "ollama: no Mistral yet. Run once (~4 GB): ollama pull mistral"
fi
