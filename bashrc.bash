#!/bin/bash

# Find the project root directory assuming this script file lives directly inside it.
COMA_PROJECT_ROOT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
export COMA_PROJECT_ROOT_DIR

# Add main and plugin code to PYTHONPATH.
if [ -z "$PYTHONPATH" ]
then
  export PYTHONPATH="$COMA_PROJECT_ROOT_DIR"/src
else
  export PYTHONPATH=$PYTHONPATH:"$COMA_PROJECT_ROOT_DIR"/src
fi

# Library path (needed only for Pycharm because of seeming bug).
# NOTE: Change your python version if needed.
PY_VERSION=python3.10
export PY_VERSION
export PYTHONPATH=$PYTHONPATH:"$COMA_PROJECT_ROOT_DIR"/.venv/lib/"$PY_VERSION"/site-packages

# Environment variables for launching without commands and configs.
export COMA_DEFAULT_CONFIG_DIR="$COMA_PROJECT_ROOT_DIR"/launch
export COMA_DEFAULT_COMMAND="test.launch"

# Create the launch config directory.
mkdir -p "$COMA_DEFAULT_CONFIG_DIR"

# Alias for program entry.
launch () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  "$PY_VERSION" "$COMA_PROJECT_ROOT_DIR"/src/main.py "$@"
  popd > /dev/null || exit
}
export -f launch

# Basic terminal auto-complete.
complete -W "
infer
test.launch
" launch

launch-dummy-infer () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  bash dummy/infer.bash "$@"
  popd > /dev/null || exit
}
export -f launch-dummy-infer

launch-dummy-infer-all-accord () {
  START_TIME=$(date +%s)
  for slice in F0 F1 AF1 F2 AF2 F3 AF3 F4 AF4 F5 AF5 ; do
    echo active_data_slice="$slice"
    launch-dummy-infer -d ACCORD -s "$slice" "$@"
  done
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}
