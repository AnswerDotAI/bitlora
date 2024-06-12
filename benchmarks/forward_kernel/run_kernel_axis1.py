import torch
import triton
import triton.language as tl
import triton_util as tu
from kernel_axis1_clean import forward_kernel
from hqq.core.bitpack import BitPack
from fastcore.script import call_parse, Param

dtype = torch.float16

def rand(*shapes): return torch.randn(*shapes, dtype=dtype, device='cuda')
def zeros(*shapes): return torch.zeros(*shapes, dtype=dtype, device='cuda')

def _quant(data, gz, axis, bits, packed):
    assert data.numel() % gz == 0, f'group_size {gz} must divide data (numel = {data.numel()})'
    assert axis in (0, 1), 'pass axis == 0 or 1'
    data = data.float().reshape(-1, gz) if axis == 1 else data.float().reshape(gz, -1)
    min_, max_ = data.min(axis=axis, keepdim=True).values, data.max(axis=axis, keepdim=True).values
    scale = (2**bits - 1) / (max_ - min_)
    zero = -min_ * scale
    data = (data * scale + zero).round()
    if packed: data = BitPack.pack_3bit_32(data)
    return data, zero, 1 / scale

def quant(data, gz, gz2, bits=3):
    qdata, zero, scale = _quant(data, gz, axis=1, bits=bits, packed=True)
    qzero, zzero, zscale = _quant(zero, gz2, axis=0, bits=bits, packed=False)
    qscale, szero, sscale = _quant(scale, gz2, axis=0, bits=bits, packed=False)
    return qdata, qzero, qscale, zzero, zscale, szero, sscale

@call_parse()
def main(
    b=4096, # Batch size
    m=4096, # Output dimension
    n=4096, # Input dimension
    r=96,   # Lora dimension
    gz=64,  # hqq group size 
    gz2=128,# hqq group size for quanting zero/scale
    bsb=16, # block size for batch dimension
    bsm=16, # block size for output dimension
    bsr=16, # block size for input dimension
    bsn=16  # block size for lora dimension
):
    assert n%gz == 0 and m%gz2== 0, f'group_size ({gz}) needs to divide n ({n}); and group_size_2 ({gz2}) needs to divide m ({m})'

    X, A, B = rand(b, n), rand(r, n) * 0.1, rand(m, r) * 0.1
    mag = rand(m).abs()
    α = rand(1).item()
    W = rand(m, n)
    W_qp, zero_q, scale_q, zzero, zscale, szero, sscale = quant(W, gz=gz, gz2=gz2)
    ng2 = m * n // (gz * gz2)
    assert (scale_q.shape, zero_q.shape, sscale.shape, szero.shape, zscale.shape, zzero.shape) == ((gz2, ng2), (gz2, ng2), (1, ng2), (1, ng2), (1, ng2), (1, ng2))

    Y = zeros(b, m)
    W_tmp = zeros(bsm, n)
    tu.assert_tensors_gpu_ready(Y, X, A, B, W_qp, mag, scale_q, zero_q, sscale, szero, zscale, zzero, W_tmp)
    grid = (tu.cdiv(b, bsb), tu.cdiv(m, bsm))

    forward_kernel[grid](
        X, A, B, W_qp, scale_q, zero_q,  # input matrices
        mag, sscale, szero, zscale, zzero, # input vectors
        α, # input scalars
        Y, # output matrix
        b, m, r, n, # dimensions
        gz, gz2, 10, # grouping / packing configs
        bsb, bsm, bsr, bsn, # block sizes
        W_tmp, # intermediate matrices
    )
