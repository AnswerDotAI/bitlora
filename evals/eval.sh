#!/bin/bash

output_dir="." # output_dir defaults to current directory

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output_dir)
            output_dir="$2"
            shift 2
            ;;
        *)    # first argument is model_dir
            model_dir="$1"
            shift
            ;;
    esac
done

if [[ -z "$model_dir" ]]; then
    echo "You need to pass model directory as the first argument."
    exit 1
fi

if ! command -v lm_eval &> /dev/null; then
    echo "lm-eval harness doesn't seem to be installed, but is needed. Should I install it into the current directory? (Y/n)"
    read -r confirm
    if [[ $confirm =~ ^[Yy]$ ]] || [[ -z $confirm ]]; then
        if [ ! -f "dharma2.yaml" ]; then
            echo "Please put the file 'dharma2.yaml' into the current directory. During installation it will be copied into the lm-eval-harness repo to define the dharma2 task."
            exit 1
        fi
        
        echo -n "Cloning lm-eval-harness repo ..."
        git clone https://github.com/EleutherAI/lm-evaluation-harness -q
        echo " done ✓"

        echo -n "Installing lm-eval-harness ..."
        pip install -e lm-evaluation-harness -qq
        echo " done ✓"

        task_dir="lm-evaluation-harness/lm_eval/tasks"
        mkdir -p "${task_dir}/dharma2"
        mv dharma2.yaml "${task_dir}/dharma2"
        echo "Moved dharma2.yaml into '${task_dir}/dharma2/' ✓"
    else
        echo "Installation aborted by you."
        exit 1
    fi
fi

echo "Evaling ${model_dir} on dharma2 with bs = auto"
# Run lm_eval with the model directory parameter
lm_eval --model hf \
        --model_args pretrained=${model_dir} \
        --tasks dharma2 \
        --device cuda:0 \
        --batch_size auto \
        --output_path ${output_dir}

echo "Result saved to '${output_dir}/result.json'  ✓"
