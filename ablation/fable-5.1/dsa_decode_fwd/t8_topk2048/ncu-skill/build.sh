#!/bin/bash
# Build cubins for B200 (sm_100a).  Usage: ./build.sh [KEYS_PER_CTA ...]   (env NVCC_EXTRA, SUFFIX, DEFS)
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
for S in "${@:-64}"; do
  $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -cubin ${NVCC_EXTRA} ${DEFS} -DKEYS_PER_CTA=$S -o dsa_s${S}${SUFFIX}.cubin dsa_kernel.cu
  echo built dsa_s${S}${SUFFIX}.cubin
done
