#!/bin/bash

while getopts ":d:rs:t:" opt; do
  case ${opt} in
    d) dataset="${OPTARG}" ;;
    r) reason=true ;;
    s) slice="${OPTARG}" ;;
    t) transform="${OPTARG}" ;;
    :)
      echo "Option -${OPTARG} requires an argument."
      exit 1
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      echo "If this is an option to pass to Coma, use '--' to delineate."
      exit 1
      ;;
  esac
done

shift "$(( OPTIND - 1 ))"

if [ -z "$dataset" ]
then
  echo "Missing Dataset Name. Use the '-d' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi
if [ -z "$transform" ]
then
  echo "Missing Transform Name. Use the '-t' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi
if [ -z "$slice" ]
then
  echo "Missing Data Slice Name. Use the '-s' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi

# Remove any lone "--" from $@
for arg
do
  shift
  [ "$arg" = "--" ] && continue
  set -- "$@" "$arg"
done

# Set the dummy llm according to the base_transform to reduce the number of variants.
# The default is ZERO_COT unless a better-fitting alternative exists.
case "$transform" in
  PLAIN) llm=plain ;;
  *) llm=zero_cot ;;
esac
if [ "$reason" ]
then
  if [ "$transform" != "PLAIN" ]
  then
    llm="$llm"_reason
  fi
fi

launch infer active_dataset="$dataset" active_data_slice="$slice" \
  active_llm="$llm" active_llm_implementation=DUMMY \
  active_transform="$transform" "$@" --dummy-path dummy/"$llm".yaml
