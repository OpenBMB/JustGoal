#!/bin/bash
cd /root/fa4work
nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a --shared -Xcompiler -fPIC -lineinfo -Xptxas -v -o libfa.so fa_fwd_b200.cu -lcuda "$@" 2>&1 | grep -E "error|registers|spill" 
echo BUILD_DONE
