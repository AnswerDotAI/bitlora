#!/bin/bash

py_file=${1%.py} # extract first argument and remove ".py" if present

echo "🔍 Profiling ${py_file}.py"

ncu --set full --import-source yes -f -o profiler_logs/$py_file --page details --target-processes all python3 $py_file.py

echo "🔍 Done ✅"
