#!/bin/zsh
set -euo pipefail

if (( $# < 1 )); then
  print -u2 "usage: $0 PROFILE [cs336-train overrides...]"
  exit 2
fi

script_dir=${0:A:h}
repo_dir=${script_dir:h}
assignment_dir=$repo_dir/assignments/assignment1-basics
profile_name=$1
shift

cd $assignment_dir
exec caffeinate -i uv run cs336-train \
  --config training_profiles.toml \
  --profile "$profile_name" \
  "$@"
