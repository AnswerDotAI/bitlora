#!/bin/bash

echo "Evaling phi-1.5 (1.3b) on hellaswag with bs = auto"

lm_eval --model hf \
    --model_args pretrained=microsoft/phi-1_5 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size auto
