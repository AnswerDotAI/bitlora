#!/bin/bash

echo "Evaling phi-1.5 (1.3b) on dharma2 with bs = auto"

lm_eval --model hf \
    --model_args pretrained=local_phi_1_5 \
    --tasks dharma2 \
    --device cuda:0 \
    --batch_size auto
