# CUDA DynamicSparseAttention grouped decode

`UseGroupedDecodeAttention` enables the split-K grouped decode attention kernel
for decode and multi-row speculative verification (`1 <= sequence_length <= 8`)
across both floating-point (`FP16`/`BF16`) and quantized (`INT8`) KV caches,
with `max_splits = 32` for quantized KV caches.

The INT8 grouped-decode and vector-staging reference tests cover one through
four query rows (`sequence_length in {1, 2, 3, 4}`).
