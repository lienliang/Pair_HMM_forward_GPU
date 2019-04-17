#!/bin/bash
nvcc -o aligner forward_gpu.cu -std=c++11 -Wno-deprecated-gpu-targets