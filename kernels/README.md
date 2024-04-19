# bitlora (WIP)

Custom qlora kernels including 1/2-bit quantization.

## Building and Running

Make sure you have `nvcc` and `cmake` installed.

To run:

```
make run
```

Alternatively to continuously rebuild/run with code edits:

```
make watch
```

## Files

- `run_bitlora.cu` - main(), entrypoint for launching kernels
- `kernels.cuh` - kernel definitions
- `kernel_toos.cuh` - convenience utilities and debugging functions.
