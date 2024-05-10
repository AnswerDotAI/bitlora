import os
import timeit
import torch
from torch.utils.cpp_extension import load_inline, load

from torch.nn import Linear
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
from dora import HQQDORA, MagnitudeLayer, DORALayer

M = 64
K = 256
N = 96
GS = 8
R = 32  # lora rank
NGROUPS = M * K // GS


def summarize(tensor: torch.Tensor):
    print("Shape:", tensor.shape)
    print("Data type:", tensor.dtype)
    print("Total elements:", tensor.numel())
    print("Minimum value:", tensor.min().item())
    print("Maximum value:", tensor.max().item())
    print(
        "Sample values:", tensor.flatten()[torch.randint(tensor.numel(), (5,))].tolist()
    )
    # show five sample values from random indices


def kernel_standalone():
    print("Initializing values...")
    x = torch.randn(M, K, dtype=torch.float).cuda()
    out = torch.zeros(M * N).cuda()
    wq = torch.randint(100, (M, ((K + 9) // 10)), dtype=torch.int32).cuda()
    loraA = torch.randn(M * R).cuda()
    loraB = torch.randn(R * N).cuda()
    loraA = qdora.dora_layer.lora_A.weight.data
    loraB = qdora.dora_layer.lora_B.weight.data
    z = torch.randn(M * K // GS).cuda()
    s = torch.randn(M * K // GS).cuda()
    print("Invoking kernel...")
    module.qdora(wq, z, s, x, out, M, K, N, GS, loraA, loraB, R, dora_scale)
    print(out)


# TODO(avh): these are for my environment, don't leave these hard coded
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["MAX_JOBS"] = "40" # for cpp_extension.load()

if __name__ == "__main__":
    sources = ["bindings.cu"]
    try:
        begin = timeit.default_timer()
        module = load(
            name="qdora",
            sources=sources,
            with_cuda=True,
            verbose=True,
            extra_cuda_cflags=["-O0", "-arch=sm_86"],
        )
        elapsed = timeit.default_timer() - begin
        print(f"Elapsed time: {elapsed:.1f}")
    except Exception as e:
        print(f"Exception while loading {sources}: {e}")

    # 10 values per 32 bits
    # torch doesn't have a uint32_t type, so we use int32_t for size
    # don't operate on the values except as a bit stream.
    x = torch.randn(M, K, dtype=torch.float).cuda()

    base_layer_uq = Linear(K, N, bias=False)
    base_layer_uq.to("cuda")
    print(base_layer_uq.weight.data)
    n_bits = 3

    # Note:
    # view_as_float makes base_layer.W_q look like a float even though the underlying tensor is quantized
    # When it is false, W_q is a packed quantized tensor
    # In [247]: base_layer.W_q.shape
    # Out[247]: torch.Size([7, 384])
    # In [248]: base_layer_uq.weight.data.shape
    # Out[248]: torch.Size([96, 256])

    quant_config = BaseQuantizeConfig(
        nbits=int(n_bits),
        group_size=64,
        quant_zero=False,
        quant_scale=False,
        offload_meta=True,
        view_as_float=False,
    )
    HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)

    print(f"x.dtype {x.dtype}")
    base_layer = HQQLinear(base_layer_uq, quant_config, compute_dtype=x.dtype)
    y = base_layer.forward(x)

    dora_scale = (
        base_layer_uq.weight.data.norm(p=2, dim=1).to(dtype=None).to("cuda")
    )  # see train.py code

    # Force dora_scale to be 1 for debugging
    # TODO(avh): remove after debugging
    # dora_scale

    base_layer.dora_scale = dora_scale
    qdora = HQQDORA(base_layer, R)

    out_ref = qdora(x)

    print(f"Result: {out_ref}")

    loraA = qdora.dora_layer.lora_A.weight.data
    loraB = qdora.dora_layer.lora_B.weight.data
    z = base_layer.meta["zero_scale"][0, 0, :].cuda()
    s = base_layer.meta["zero_scale"][1, 0, :].cuda()

    # zero point and scale
    # In [61]: base_layer.meta.keys()
    # Out[61]: dict_keys(['nbits', 'group_size', 'shape', 'axis', 'packing', 'unpack_view_dtype', 'view_as_float', 'quant_scale', 'quant_zero', 'meta_zero', 'meta_scale', 'compute_dtype', 'zero_scale'])
    # Is scale quantized, is zero quntized? (yes and yes)
    # In [139]: base_layer.meta['quant_scale']
    # Out[139]: True
    # In [140]: base_layer.meta['quant_zero']
    # Out[140]: True
    # Is this right?
    # In [128]: summarize(base_layer.meta['zero_scale'])
    # Shape: torch.Size([2, 128, 3])
    #                    |  |    ^ Groups
    #                    |   +---- Elements within group
    #                    +-------- [0] = (z) zero point (quantized), [1] = (s) scale (quantized)
    # Data type: torch.uint8
    # Total elements: 768
    # Minimum value: 0
    # Maximum value: 255
    # Sample values: [25, 81, 54, 34, 35]

    # TODO(avh): for now kernel only accepts float32
    assert x.dtype == torch.float32
    w_q = base_layer.W_q.data
    # TODO(avh): what's the relationship between N * K and w_q.shape?
    # It's roughly N * K // 10 ~ prod(w_q.shape) add assert when we have precise relationship
    out = torch.zeros(M * N).cuda()
    module.qdora(w_q, z, s, x, out, M, K, N, GS, loraA, loraB, R, dora_scale)
    out = out.reshape(out_ref.shape)
    print("\n\nReference Implementation:")
    print(out_ref)
    print("\n\nKernel Implementation:")
    print(out)
