`nbits_v_quality_hqq-qdora.ipynb` investigates how the number of bits used in hqq-qdora impacts the quality of the finetuned modal.

The `trn-*.sh` scripts were used to run the experiments. Note that currently, the experiments are logged to the w&b of Umer.
To finetune with hqq-qdora ...
- a single `llama2-7b` model with `n` bits, use `trn_qdora_llama2.sh --bits <n>`
- a single `llama3-8b` model with `n` bits, use `trn_qdora_llama3.sh --bits <n>` 
- a `llama2-7b` model for `4`, `3`, `2`, `1` bits each, use `trn_qdora_llama2_different_bits.sh`
- a `llama3-8b` model for `4`, `3`, `2`, `1` bits each, use `trn_qdora_llama3_different_bits.sh`

For each, use `--small` to test if training starts, as this will only use 100 samples.
