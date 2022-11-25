#!/bin/sh

# Go to "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html"
# Add Compile Options to Makefile
# Make

# export MKL_NUM_THREADS=4
# export OMP_NUM_THREADS=4
# xport MKL_THREAD_MODEL=OMP

make clean && make && ./lstm
