#!/bin/bash

py_file=${1%.py} # extract first argument and remove ".py" if present

mkdir -p profiler_logs # create dir 'profiler_logs' is not exists

echo "🔍 Profiling ${py_file}.py"
echo "(profiling logs will be written to 'profiler_logs/${py_file}.ncu-rep'.)"

ncu --set full --import-source yes -f -o profiler_logs/$py_file --page details --target-processes all python3 $py_file.py

echo "🔍 Done ✅"
