import copy
import os
from datetime import date
from pprint import pprint
import pdb
import sys
from typing import List, Set
import timeit

import bitsandbytes as bnb

from composer import functional as F
from composer import Trainer
from composer.models.base import ComposerModel
from composer.metrics import CrossEntropy
from composer.models.huggingface import HuggingFaceModel

import pandas as pd

from torch.optim.lr_scheduler import OneCycleLR

from dataclasses import dataclass
from datasets import load_dataset
import transformers as tf
import torch
import torch.nn as nn
import hqq.core.quantize as hqq
from loguru import logger


def fetch_gemma_model(model_name: str, device_map={"": 0}):
    """Fetch the unquantized version of the Gemma model (f32)"""
    if not ("HF_TOKEN" in os.environ.keys()):
        print("shell variable HF_TOKEN needs to be set to authentication key")
        sys.exit(0)
    tokenizer = tf.AutoTokenizer.from_pretrained(
        model_name, token=os.environ["HF_TOKEN"]
    )
    model = tf.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        token=os.environ["HF_TOKEN"],
    )
    return model, tokenizer


# BitLora Model Definitions


class BitLoraBlock(nn.Module):
    """BitLoraBlock is a combined base layer quantized module + two linear lora layers."""

    def __init__(
        self, lora_a: nn.Linear, lora_b: nn.Linear, base_layer: nn.Module, alpha: float
    ):
        super(BitLoraBlock, self).__init__()
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.base_layer = base_layer
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        # using hqq_base as the pretrained path, apply lora_a and lora_b
        y = self.base_layer(x)
        x = self.lora_a(x)
        x = self.lora_b(x)
        return x + y


class BitGemma(ComposerModel):
    """Wraps model for Mosaic Composer training."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = model.config

    def loss(self, outputs, batch):
        # TODO
        return None

    def forward(self, batch):
        x, _ = batch
        y = self.model.generate(x)
        # TODO
        return None


def bitlora_block(
    layer: nn.Linear = None,
    lora_size: int = 32,
    quant_config=hqq.BaseQuantizeConfig(nbits=4, group_size=64),
    compute_dtype=torch.float32,
    alpha=1e-2,
    device="cuda",
    del_orig=True,
):
    """Factory function for BitLoraBlock"""
    lora_a = nn.Linear(layer.in_features, lora_size, bias=False, device=device)
    lora_b = nn.Linear(lora_size, layer.out_features, bias=False, device=device)
    # Zero initialie so by default this should produce the same output as the
    # quantized model
    nn.init.constant_(lora_a.weight, 0.0)
    nn.init.constant_(lora_b.weight, 0.0)
    base_layer = hqq.HQQLinear(
        layer,
        quant_config=quant_config,
        compute_dtype=compute_dtype,
        device=device,
        initialize=True,
        del_orig=del_orig,
    )
    result = BitLoraBlock(lora_a, lora_b, base_layer, alpha)
    return result


def make_bitlora_model(
    model: nn.Module,
    target_modules: Set[str],
    lora_size: int,
    quant_config,
    compute_dtype=torch.float32,
    alpha=1e-2,
    device="cuda",
):
    """Recursively traverse the model for each nn.Module child, if the child type
    is Linear and child name is in target_modules, replace it with BitLoraBlock.
    Modifies model in-place."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name in target_modules:
            setattr(
                model,
                name,
                bitlora_block(
                    child,
                    lora_size,
                    quant_config,
                    compute_dtype,
                    alpha,
                    device,
                    del_orig=True,
                ),
            )
        else:
            make_bitlora_model(child, target_modules, lora_size, quant_config)
    return model


def freeze(model: nn.Module, training_modules: Set[str]):
    """Freeze all modules that are not in target_modules."""
    for name, child in model.named_children():
        if name not in training_modules:
            logger.info("Freezing module: {}", name)
            child.requires_grad = False
        else:
            child.requires_grad = True
            logger.info("Trainable module: {}", name)
        freeze(child, training_modules)
    return model


def make_hqq_model(
    model: nn.Module,
    target_modules: Set[str],
    quant_config,
    compute_dtype=torch.float32,
    device="cuda",
):
    """Recursively traverse the model for each nn.Module child, if the child type
    is Linear and child name is in target_modules, replace it with HQQLinear.
    Modifies model in-place."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name in target_modules:
            setattr(
                model,
                name,
                hqq.HQQLinear(
                    child,
                    quant_config=quant_config,
                    compute_dtype=compute_dtype,
                    device=device,
                    initialize=True,
                    del_orig=True,
                ),
            )
        else:
            make_hqq_model(child, target_modules, quant_config)
    return model


def test_generation(model, tokenizer, quote_prefix: str, max_new_tokens=80):
    prompt = '{\n  "quote" : "' + quote_prefix
    device = model.device
    state = {}
    # time tokenizer and put tokenized tensor in state
    duration = timeit.timeit(
        stmt=lambda: state.update(
            {"input_tokens": tokenizer(prompt, return_tensors="pt").to(device)}
        ),
        number=1,
    )
    logger.info(f"Tokenization time: {duration:.3f} seconds")
    # Time generation and put generated tokens in state
    duration = timeit.timeit(
        stmt=lambda: state.update(
            {
                "output_tokens": model.generate(
                    **state["input_tokens"], max_new_tokens=max_new_tokens
                )
            }
        ),
        number=1,
    )
    logger.info(
        "Output: \n"
        + tokenizer.decode(state["output_tokens"][0], skip_special_tokens=True)
    )


def check_weights(model: nn.Module):
    """Sanity check - manually inspect some weights in layer 0."""
    embeddingT = list(model.children())[1]
    embedding, layers, final_norm = list(list(model.children())[0].children())
    layer0 = layers[0]
    attn, mlp, rmsnorm0, rmsnorm1 = list(layer0.children())
    weight_matrix = mlp.gate_proj.weight.data
    return mlp.gate_proj, weight_matrix


def check_bitlora_weights(model: nn.Module):
    """Sanity check - manually inspect some bitlora weights in layer 0."""
    embeddingT = list(model.children())[1]
    embedding, layers, final_norm = list(list(model.children())[0].children())
    layer0 = layers[0]
    attn, mlp, rmsnorm0, rmsnorm1 = list(layer0.children())
    weight_matrix = list(attn.q_proj.base_layer.parameters())[0].data

    lora_a = attn.q_proj.lora_a.weight.data
    lora_b = attn.q_proj.lora_b.weight.data
    hqq_base = attn.q_proj.base_layer

    return lora_a, lora_b, hqq_base


def reset_environment():
    """Used for interactive repl sessions, wipe the environment."""
    torch.cuda.empty_cache()
    for name in dir():
        if not name.startswith("_"):
            del globals()[name]


def formatting_func(example):
    # strip stylized quotation marks, to be replaced by a plain quote "
    example_stripped = example["quote"][1:-1]
    text = (
        f"{{"
        f'\n  "quote" : "{example_stripped}",\n'
        f"  \"author\" : \"{example['author']}\",\n"
        f'  "dataset_date": "{str(date.today())}"\n'
        f"}}"
    )
    return text


def prep_data():
    # From compose tutorial: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
    rd_ds = load_dataset("xiyuez/red-dot-design-award-product-description")
    rd_df = pd.DataFrame(rd_ds['train'])
    rd_df['instruction'] = 'Create a detailed description for the following product: '+ rd_df['product']+', belonging to category: '+ rd_df['category']
    rd_df = rd_df[['instruction', 'description']]
    rd_df_sample = rd_df.sample(n=5000, random_state=42)
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:

    {}

    ### Response:\n"""
    rd_df_sample['prompt'] = rd_df_sample["instruction"].apply(lambda x: template.format(x))
    rd_df_sample.rename(columns={'description': 'response'}, inplace=True)
    rd_df_sample['response'] = rd_df_sample['response'] + "\n### End"
    rd_df_sample = rd_df_sample[['prompt', 'response']]
    rd_df_sample['text'] = rd_df_sample["prompt"] + rd_df_sample["response"]
    rd_df_sample.drop(columns=['prompt', 'response'], inplace=True)
    return rd_ds, rd_df_sample, rd_df



if __name__ == "__main__":
    reset_environment()

    PARAMS = {
        "lora_size": 32,
        "dataset": "Abirate/english_quotes",
    }
    target_modules = set(
        [
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    logger.info("Fetching base model")
    gemma_base, tokenizer = fetch_gemma_model("google/gemma-2b")

    # BitLora Model Setup
    quant_config = hqq.BaseQuantizeConfig(nbits=4, group_size=64)
    logger.info("Creating bitlora model")
    bitlora = make_bitlora_model(
        gemma_base,
        target_modules,
        PARAMS["lora_size"],
        quant_config,
        compute_dtype=torch.float32,
        alpha=1e-2,
        device="cuda",
    )
    logger.info("Checking bitlora weights")
    lora_a, lora_b, hqq_base = check_bitlora_weights(bitlora)
    pprint(lora_a)
    pprint(lora_b)
    pprint(hqq_base)

    # Poke at HQQ params
    sd = bitlora.base_model.layers[0].mlp.gate_proj.base_layer.state_dict()
    pprint(sd)
    pprint(sd['meta']['scale'].shape) # torch.Size([1, 524288])
    pprint(sd['meta']['zero_q'].shape) # torch.Size([1, 524288])
    pprint(sd['W_q'].shape)

    logger.info("Freezing model and testing generation")
    frozen = freeze(bitlora, set(["lora_a", "lora_b"]))
    test_generation(bitlora, tokenizer, "it was the best of times it was")

    # Data Setup
    logger.info("Preparing dataset")
    data = load_dataset(path=PARAMS["dataset"])
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    dataloader = torch.utils.data.DataLoader(data)

    # Trainer setup (WIP)
    train_model = BitGemma(frozen)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-3)
    batch_size = 50
    steps_per_epoch = len(data['train']) // batch_size
    scheduler = OneCycleLR(
        optimizer,
        0.1,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
    )
