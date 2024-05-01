#!/bin/bash

dataset_samples=10000
for arg in "$@"; do
    [ "$arg" == "--small" ] && dataset_samples=100
done

echo "Will train llama3 8b with hqq_dora on ${dataset_samples} orca-math samples for 4 / 3 / 2 / 1 bits each"


for bits in 4 3 2 1
do
    echo "🔥🔥 ${bits} bits 🔥🔥 Training llama3 8b with hqq_dora on ${dataset_samples} orca-math samples with ${bits} bits"
    
    python train.py \
        --model_name meta-llama/Meta-Llama-3-8B \
        --train_type hqq_dora \
        --n_bits $bits \
        --precision bf16 \
        --dataset orca_math \
        --dataset_samples $dataset_samples \
        --batch_size 4 \
        --context_length 2048 \
        --gradient_accumulation_steps 2 \
        --use_gradient_checkpointing true \
        --reentrant_checkpointing true \
        --use_cpu_offload false \
        --use_activation_cpu_offload false \
        --log_to wandb \
        --project_name "llama3-8b-quant-ft-${bits}bit" \
        --save_model true \
        --output_dir "../models/llama3-8b-orca-math-${dataset_samples}-hqq-qdora-${bits}bits"
done
