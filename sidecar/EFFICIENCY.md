# Do not lose this pattern

The graph trainer got slow again when a new loss walked the batch in Python on
the GPU thread, and when the batch dropped from 256 to 32. Keep the pattern
below. It is the same one the oscillator trainer uses.

- One matmul covers the whole batch. A Python loop over stories belongs on the
  prefetch thread, never inside the GPU step.
- The prefetch thread packs the next batch while the GPU runs. The queue holds
  two batches. Pack tokens and beat readings there.
- Copy to the device with a blocking `.to(device)`. On MPS, `pin_memory` and
  `non_blocking=True` hand the step garbage token ids.
- `--batch-size` is per device and stays 256 unless there is a measured reason
  to change it. On 10,000 stories that is about 40 steps an epoch. Batch 32 is
  about 313 steps and the epoch feels slow even when each step is smaller.
- Sync the GPU on the 15-second log line. Do not `.item()` or `float()` a
  training tensor on every step.
- Do not `torch.compile` this word loop on this Mac. The backward does not
  build and the process hangs. Do not add mlx to pyproject for the graph.
- Train on MPS or CUDA. If neither exists, stop. Do not fall back to CPU.
