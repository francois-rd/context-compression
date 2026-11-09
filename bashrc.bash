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
  if ! "$PY_VERSION" "$COMA_PROJECT_ROOT_DIR"/src/main.py "$@" ; then return 1 ; fi
  popd > /dev/null || exit
}
export -f launch

# Basic terminal auto-complete.
complete -W "
infer
postprocess
test.launch
" launch

_launch-infer () {
  local OPTIND
  local opt implementation folder
  while getopts ":i:" opt; do
    case ${opt} in
      i) implementation="${OPTARG}" ;;
      :)
        echo "Option -${OPTARG} requires an argument."
        return 1
        ;;
      *) echo skipping "${OPTARG}" ;;
    esac
  done
  shift "$(( OPTIND - 1 ))"
  case "$implementation" in
    DUMMY) folder=dummy ;;
    OPENAI) folder=openai ;;
    *)
      echo "Invalid implementation: '$implementation'. Choose from: DUMMY, OPENAI."
      return 1
      ;;
  esac
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  if ! bash "$folder"/infer.bash "$@" ; then return 1 ; fi
  popd > /dev/null || exit
}
export -f _launch-infer

_launch-infer-baseline-transforms () {
  for transform in PLAIN ZERO_COT ; do
    echo active_transform="$transform"
    _launch-infer "$1" "$2" -- -t "$transform" "${@:3}"
  done
  for keep_p in 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 ; do
    for unit in NEWLINE TOKEN ; do
      echo active_manipulation=RANDOM keep_p="$keep_p" unit="$unit"
      if ! \
        _launch-infer "$1" "$2" -- -t FORCED_COT "${@:3}" \
           forced_cot.manipulation=TOKEN_SKIP \
           token_skip.random=true token_skip.keep_p="$keep_p" token_skip.unit="$unit"
      then
        return 1
      fi
    done
  done
}
export -f _launch-infer-baseline-transforms

_launch-infer-token-skip () {
  for keep_p in 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 ; do
    for unit in NEWLINE TOKEN ; do
      echo active_manipulation=TOKEN_SKIP keep_p="$keep_p" unit="$unit"
      if ! \
        _launch-infer "$1" "$2" -- -t FORCED_COT "${@:3}" \
          forced_cot.manipulation=TOKEN_SKIP \
          token_skip.random=false token_skip.keep_p="$keep_p" token_skip.unit="$unit"
      then
        return 1
      fi
    done
  done
}
export -f _launch-infer-token-skip

_launch-infer-tail-pruning-base () {
  for keep_p in 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 ; do
    # shellcheck disable=SC2043
    for unit in NEWLINE ; do
      echo active_manipulation=TAIL_PRUNING keep_p="$keep_p" unit="$unit"
      if ! \
        _launch-infer "$1" "$2" -- -t FORCED_COT "${@:3}" \
          forced_cot.manipulation=TAIL_PRUNING \
          tail_pruning.keep_p="$keep_p" tail_pruning.unit="$unit"
      then
        return 1
      fi
    done
  done
}
export -f _launch-infer-tail-pruning-base

_launch-infer-tail-pruning-peak () {
  for keep_p in 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 ; do
    # shellcheck disable=SC2043
    for exclusivity_window in 10 ; do
      for frequency_scaler in 1.0 ; do
        echo keep_p="$keep_p" window="$exclusivity_window" scaler="$frequency_scaler"
        if ! \
          _launch-infer "$1" "$2" -- -t FORCED_COT "${@:3}" \
            forced_cot.manipulation=TAIL_PRUNING \
            tail_pruning.keep_p="$keep_p" tail_pruning.unit=PEAK \
            tail_pruning.exclusivity_window="$exclusivity_window" \
            tail_pruning.frequency_scaler="$frequency_scaler"
        then
          return 1
        fi
      done
    done
  done
}
export -f _launch-infer-tail-pruning-peak

_launch-infer-all-transform-manipulations () {
  START_TIME=$(date +%s)
  if ! _launch-infer-baseline-transforms "$@" ; then return 1 ; fi
  if ! _launch-infer-token-skip "$@" ; then return 1 ; fi
  if ! _launch-infer-tail-pruning-base "$@" ; then return 1 ; fi
  if ! _launch-infer-tail-pruning-peak "$@" ; then return 1 ; fi
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}
export -f _launch-infer-all-transform-manipulations

_launch-infer-accord () {
  for slice in F0 F1 AF1 F2 AF2 F3 AF3 F4 AF4 F5 AF5 ; do
    echo active_data_slice="$slice"
    if ! \
      _launch-infer-all-transform-manipulations "$1" "$2" -d ACCORD -s "$slice" "${@:3}"
    then
      return 1
    fi
  done
}
export -f _launch-infer-accord

launch-infer-accord-dummy () {
  _launch-infer-accord -i DUMMY "$@"
}

launch-infer-accord-openai () {
  _launch-infer-accord -i OPENAI "$@"
}
