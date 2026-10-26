#!/bin/bash
# Build the from-scratch FA-forward kernel as a shared library (sm_100a / B200).
set -e
cd "$(dirname "$0")"
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -shared -Xcompiler -fPIC -o libfa.so fa_fwd.cu
echo "built libfa.so"
# tests / benchmarks (run from src/): python3 test_fa.py 1024 32768 ; python3 bench_cmp.py 2 ; python3 cold_probe.py
