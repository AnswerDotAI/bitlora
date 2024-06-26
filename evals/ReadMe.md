- We're using [dharma2 benchmark](https://huggingface.co/datasets/pharaouk/dharma-2) to quickly eval models, as its wide (many different tasks) and small (300 total questions)
- We're using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as framework, and have verified:
	- we're using the framework correctly (can reproduce phi1.5 metrics from its paper)
	- larger models (llama3-8b) perform better than smaller (phi1.5 which is 1.3b) - this is a plausibility check
  - we can evaluate local models

- Use `./eval.sh <model_dir> -o <output_dir>` to eval a model. The result is saved to `<output_dir>/result.json`.
