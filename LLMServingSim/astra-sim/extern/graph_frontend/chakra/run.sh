#!/bin/bash

python3 -m chakra.src.converter.converter LLM \
    --input ../../../inputs/trace/examples/llm_hybrid.txt \
    --output ../../../inputs/workload/llm \
    --num-npus 8
