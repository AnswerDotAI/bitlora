Austin profiled fsdp_qdora for hqq, and shared the files
- `bs_8_profiler_stacks_1_20240506_134446.txt`
- `bs_8_profiler_stacks_0_20240506_134448.txt`
- `bs_2_profiler_stacks_0.txt`

With [speedscope](https://www.speedscope.app/), we see that qdora fwd makes up ~half of all compute time. bwd is likely not too far off.
So the question "Will making qdora fwd / bwd faster make fsdp_qdora significantly faster?" can be answered with yes.
