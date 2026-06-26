#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/assignments/assignment1-basics/data"

mkdir -p "${DATA_DIR}"

download() {
  local url="$1"
  local dest="$2"

  if [[ -f "${dest}" ]]; then
    echo "Already exists: ${dest}"
    return
  fi

  echo "Downloading ${url}"
  curl -L --fail --continue-at - "${url}" -o "${dest}"
}

download "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
  "${DATA_DIR}/TinyStoriesV2-GPT4-train.txt"
download "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" \
  "${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt"

download "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz" \
  "${DATA_DIR}/owt_train.txt.gz"
download "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz" \
  "${DATA_DIR}/owt_valid.txt.gz"

for path in "${DATA_DIR}"/owt_*.txt.gz; do
  [[ -f "${path}" ]] || continue
  out="${path%.gz}"
  if [[ -f "${out}" ]]; then
    echo "Already exists: ${out}"
  else
    echo "Decompressing ${path}"
    gzip -dk "${path}"
  fi
done

echo "Assignment 1 data is ready in ${DATA_DIR}"

