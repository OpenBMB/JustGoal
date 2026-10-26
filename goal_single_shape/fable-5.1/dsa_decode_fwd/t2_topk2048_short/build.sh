#!/bin/bash
set -e
cd "$(dirname "$0")"
nvcc -O3 -std=c++17 -arch=sm_100a -diag-suppress 177 -cudart shared -Xcompiler -fPIC -shared dsa_kernel.cu -o libdsa.so
