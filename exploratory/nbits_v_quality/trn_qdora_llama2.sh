#!/bin/bash

# Default values
dataset_samples=10000
bits=4

# Argument parsing
for arg in "$@"
do
    case $arg in
        --small)
        dataset_samples=100
        shift # Remove --small from processing
        ;;
        --bits=*)
        bits="${arg#*=}"
        shift # Remove --n_bits=value from processing
        ;;
    esac
done

echo "Training llama2 7b with hqq_dora on ${dataset_samples} orca-math samples with ${bits} bits"

python train.py \
    --model_name meta-llama/Llama-2-7b-hf \
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
    --project_name "llama2-7b-quant-ft-${bits}bit" \
    --save_model true \
    --output_dir "../models/llama2-7b-orca-math-${dataset_samples}-hqq-qdora-${bits}bits"
