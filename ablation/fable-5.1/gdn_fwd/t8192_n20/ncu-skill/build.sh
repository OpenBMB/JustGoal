#!/bin/bash
set -e
cd "$(dirname "$0")"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_100a -Xptxas -v --shared -Xcompiler -fPIC -o libgdn.so gdn.cu "$@" 2>&1 | grep -E "error|warning|registers|spill|bytes stack" || true
ls -la libgdn.so
