#!/bin/bash
set -e
cd "$(dirname "$0")"
nvcc -arch=sm_100a -cubin -O3 -DNT=512 -DBPSM=2 -DRSEL=3 -DWIN=7 -o moe.cubin moe19.cu
