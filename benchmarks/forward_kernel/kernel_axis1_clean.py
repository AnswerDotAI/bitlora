# Cleaned version (ie without build-up and tests) of fused-hqq-qdora-fwd kernel for axis=1. See `kernel_axis1.ipynb` for version with build-up and tests.

import triton, triton.language as tl, triton_util as tu
const = tl.constexpr # abbreviations so I have to type less

pz = 10 # pack size is always 10, as we pack 10 3bits -> a 32bit

@triton.jit
def dequant_zero_or_scale(vals_ptr, zero_ptr, scale_ptr, gz: const, gz2: const, ng2: const, m: const, n: const, bsm: const, idm):
    # load vals_q, which is 2d-shape (gz2,ng2) as 1d-shape (gz2*ng2), this allows loading only the chunk we need
    vals_q = tu.load_1d(vals_ptr, sz=bsm*n//gz, n=idm, max=gz2*ng2) # quanted zero/scale values ; ~ (bsm*n//gz)
    # now, load zero/scale via modulo
    offs = tu.offset_1d(sz=bsm*n//gz, n_prev_chunks=idm) % ng2
    zero = tl.load(zero_ptr + offs, offs < ng2)  # ~ zero  of zero/scale; (bsm*n//gz)
    scale= tl.load(scale_ptr+ offs, offs < ng2)  # ~ scale of zero/scale; (bsm*n//gz)
    # dequant and return
    return (vals_q-zero)*scale  # ~ (bsm*n//gz)

@triton.jit
def load_W_q(ptr, bsm: const, idm: const, m: const, n: const, gz: const, pz: const):
    '''Load selected rows from W_qp (and all cols)'''
    offs   = tu.offset_1d(bsm, idm)  # rows of W
    offs_q = tu.offset_1d(bsm*(n//gz), idm) # rows of W_q ; note: n/gz is integer    
    npacks = (m*n//gz + 10 - 1) // 10  # = cdiv(m*n//gz, 10) = number of packed values
    offs0  = offs_q % npacks           # rows of W_qp
    shifts = 27 - 3*(offs_q // npacks) # right-shifts needed to unpack correctly, for each row    
    offs1 = tl.arange(0, gz) # all cols
    w_pq_offs = tu.offset_2d(offs0, offs1, stride0=gz)
    w_pq_mask = tu.mask_2d  (offs0, offs1, max0=m*n//gz, max1=gz)
    vals = tl.load(ptr + w_pq_offs, w_pq_mask)
    shifted_vals = vals >> shifts[:,None] & 0b111
    return shifted_vals

@triton.jit
def dequant_W(W_q, zero, scale, bsm: const, n:const):
    # W_q ~ (bsm*n//gz, gz), zero/scale ~ (bsm*n//gz)
    return ((W_q-zero[:,None])*scale[:,None]).reshape(bsm,n)

@triton.jit
def batched_matmul(
    x_ptr, w_ptr,                       # pointers
    bsb: const, bsm: const, bsn: const, # block sizes
    idb, idm,                           # block indices
    b:const, n:const,                   # matrix sizes
):
    # 1d offets along b,m,n axes
    offs_b = tu.offset_1d(bsb, idb)
    offs_m = tu.offset_1d(bsm, idm)
    offs_n = tu.offset_1d(bsn, 0)
    # 2d offsets of x, w
    offs_x = x_ptr + tu.offset_2d(offs_b, offs_n, n)
    offs_w = w_ptr + tu.offset_2d(offs_m, offs_n, n)
    # initialize and iteratively update accumulator
    acc = tl.zeros((bsb, bsm), dtype=tl.float32)
    for _ in range(0, n, bsn):
        x = tl.load(offs_x)
        w = tl.load(offs_w)
        acc += tl.dot(x, w.trans(), allow_tf32=False) # allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offsets, so next iteration loads next chunks
        offs_x += bsn
        offs_w += bsn
    return acc

@triton.jit
def batched_double_matmul(
    x_ptr, a_ptr, b_ptr,                            # pointers
    bsb: const, bsm: const, bsn: const, bsr: const, # block sizes
    idb, idm,                                       # block indices
    n, r                                            # matrix sizes
):
    # 1d offets along b,m,n,r axes
    offs_b = tu.offset_1d(bsb, idb)
    offs_m = tu.offset_1d(bsm, idm)
    offs_n = tu.offset_1d(bsn, 0)
    offs_r = tu.offset_1d(bsr, 0)
    # 2d offsets of x,a,b
    offs_x = x_ptr + tu.offset_2d(offs_b, offs_n, n)
    offs_a = a_ptr + tu.offset_2d(offs_r, offs_n, n)
    offs_b = b_ptr + tu.offset_2d(offs_m, offs_r, r)
    # initialize and iteratively update accumulator
    acc = tl.zeros((bsb, bsm), dtype=tl.float32)
    for idn in range(0, n, bsn):
        x = tl.load(offs_x) # load x chunk (axis: b, n)
        for idr in range(0, r, bsr):
            a = tl.load(offs_a) # load a chunk (axis: r,n)
            b = tl.load(offs_b) # load a chunk (axis: m,r)
            acc += tl.dot( # multiply x with a.t, cast result to fp16, then multiply with b.t 
                tl.dot(x, a.trans(), allow_tf32=False).to(tl.float16),
                b.trans(),
                allow_tf32=False
            ) # allow_tf32 must be set to False for older GPUs, otherwise won't compile
            # increase offsets, so next iteration loads next chunks
            offs_a += bsr * n # advance a by bsr rows
            offs_b += bsr     # advance b by bsr cols
        offs_a = a_ptr + tu.offset_2d(offs_r, offs_n, n) # reset advancement of a
        offs_b = b_ptr + tu.offset_2d(offs_m, offs_r, r) # reset advancement of b
        offs_a += bsn # advance a by bsn cols
        offs_x += bsn
    return acc

@triton.jit
def col_norm(
    w_ptr, a_ptr, b_ptr,                # pointers
    bsm: const, bsn: const, bsr: const, # block sizes
    idm,                                # block indices
    n, r                                # matrix sizes
):
    # 1d offets along m,n,r axes
    offs_m = tu.offset_1d(bsm, idm)
    offs_n = tu.offset_1d(bsn, 0)
    offs_r = tu.offset_1d(bsr, 0)
    # 2d offsets of x,a,b
    offs_w = w_ptr + tu.offset_2d(offs_m, offs_n, n)
    # initialize and iteratively update accumulator for colnorm, using a regular matmul to compute b@a
    acc_colnorm = tl.zeros((bsm,bsn), dtype=tl.float32)  # accumulator for colnorm
    for idb in range(0, n, bsn):
        acc_mm = tl.zeros((bsm, bsn), dtype=tl.float32) # initialize accumulator for matmul
        # initialize / reset offsets for a & b
        offs_a = a_ptr + tu.offset_2d(offs_r, offs_n, n)
        offs_b = b_ptr + tu.offset_2d(offs_m, offs_r, r)        
        for idr in range(0, r, bsr):
            a = tl.load(offs_a)
            b = tl.load(offs_b)
            acc_mm += tl.dot(b, a, allow_tf32=False) # allow_tf32 must be set to False for older GPUs, otherwise won't compile
            # increase offsets, so next iteration loads next chunks
            offs_a += bsr*n # advance a by bsr rows
            offs_b += bsr   # advance b by bsr cols
        w = tl.load(offs_w)
        # update accumulator for colnorm
        acc_colnorm += (acc_mm+w)*(acc_mm+w)
    return tl.sqrt(tl.sum(acc_colnorm, axis=1))

@triton.jit
def pointwise_mul(y, mag_ptr, colnorm, α, bsm: const, idm, m):
    mag = tu.load_1d(mag_ptr, bsm, idm, m)
    return α * y * mag[None,:] / colnorm[None,:]

def fwd_op(fwd_k, X, A, B, W_qp, scale_q, zero_q, mag, sscale, szero, zscale, zzero, α, bsb, bsm, bsr, bsn):
    (b,n), (m,r) = X.shape, B.shape
    Y = zeros(b,m)
    W_tmp = zeros(bsm,n)
    tu.assert_tensors_gpu_ready(Y, X, A, B, W_qp, mag, scale_q, zero_q, sscale, szero, zscale, zzero, W_tmp)
    grid = (cdiv(b, bsb), cdiv(m, bsm))
    print(f'Launching grid of size {grid}')
    fwd_k[grid](
        X, A, B, W_qp, scale_q, zero_q,  # input matrices
        mag, sscale, szero, zscale, zzero, # input vectors
        α, # input scalars
        Y, # output matrix
        b,m,r,n,# dimensions
        gz, gz2, pz, # grouping / packing configs
        bsb, bsm, bsr, bsn, # block sizes
        # todo umer: add? # strides
        W_tmp, # todo umer: should not need this # intermediate matrices
    )
    return Y

@triton.jit
def forward_kernel(    
    X_ptr, A_ptr, B_ptr, W_qp_ptr, scale_q_ptr, zero_q_ptr,       # input matrices
    magnitude_ptr, sscale_ptr, szero_ptr, zscale_ptr, zzero_ptr,  # input vectors
    α,                                                            # input scalars
    Y_ptr,                                                        # output matrix
    b: const, m: const, r: const, n: const,                       # dimensions
    gz: const, gz2: const, pz: const,                             # grouping / packing configs
    bsb: const, bsm: const, bsr: const, bsn: const,               # block sizes
    # todo umer: add?                                             # strides
    W_tmp_ptr, # todo umer: should not need this                  # intermediate matrices
):
    tl.static_assert(n%gz==0, f'group_size ({gz}) must divide n ({n})')
    tl.static_assert(m%gz2==0,f'group_size2 ({gz2}) must divide m ({m})')
    idb, idm = tl.program_id(0), tl.program_id(1) # block ids correspond to chunking of b and m axes
    ng:  const  = m*n//gz # number of groups
    ng2: const  = ng//gz2 # number of groups for quanting zero/scale
    # (1) dequant zero & scale
    zero  = dequant_zero_or_scale(zero_q_ptr,  zzero_ptr, zscale_ptr, gz, gz2, ng2, m, n, bsm, idm)
    scale = dequant_zero_or_scale(scale_q_ptr, szero_ptr, sscale_ptr, gz, gz2, ng2, m, n, bsm, idm)
    # (2) unpack w
    W_q = load_W_q(W_qp_ptr, bsm, idm, m, n, gz, pz)
    # (3) dequant w
    W = dequant_W(W_q, zero, scale, bsm, n)
    tu.store_full_2d(W, W_tmp_ptr, bsm, n, n) # todo umer: Remove intermediate storage in global mem
    # (4) x@w.t
    XW = batched_matmul(X_ptr, W_tmp_ptr, bsb, bsm, bsn, idb, 0, b, n) # set idm=0, because W is already an m-chunk
    # (5) x@a.t@b.t
    XAB = batched_double_matmul(X_ptr, A_ptr, B_ptr, bsb, bsm, bsn, bsr, idb, idm, n, r)    
    # (6) colnorm
    colnorm = col_norm(W_tmp_ptr, A_ptr, B_ptr, bsm, bsn, bsr, 0, n, r) # set idm=0, because W is already an m-chunk
    # (7) * α, mag, 1/colnorm    
    Y = pointwise_mul(XW + XAB, magnitude_ptr, colnorm, α, bsm, idm, m)
    # store
    tu.store_2d(Y, Y_ptr, bsb, bsm, idb, idm, b, m, m)
