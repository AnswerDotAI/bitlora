#!/bin/bash

# Default value for b
b=4
outfile=""

# Parse the named argument for --b
while [ $# -gt 0 ]; do
  case "$1" in
    --b=*)
      b="${1#*=}"
      ;;
    --outfile=*)
      outfile="${1#*=}"
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done

# Check if outfile is set
if [ -z "$outfile" ]; then
  echo "Please provide --outfile, which is where the profiling info will be saved to."
  exit 1
fi

echo "Profiling forward kernel for batch_size = $b"

ncu --target-processes=all --set full --export $outfile --page "details" --import-source yes --call-stack python run_kernel_axis1.py --b=$b
