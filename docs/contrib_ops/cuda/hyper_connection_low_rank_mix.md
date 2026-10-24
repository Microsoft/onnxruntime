# CUDA HyperConnectionLowRankMix decode tuning

`ORT_HC_DECODE_ROWS` selects the existing fused-kernel row grouping for eligible
four-row decode inputs. The default is 1 (one row per work group), which
optimizes four-row speculative verification on Hopper (`SM90+`). Set it to 2
before starting the process to use two-row work groups.

Only values 1 and 2 are supported. Other input shapes and unsupported fused
configurations retain their existing dispatch. The operator schema, output
layout, and accumulation semantics are unchanged.

`ORT_HYPER_CONNECTION_PDL` controls Hopper programmatic dependent launch
(`cudaLaunchAttributeProgrammaticStreamSerialization`) between `HcDownKernel`
and `HcUpPreKernel` on `SM90+` when weights are constant initializers. It
defaults to `1` (`true`).

The rank-320 fused-decode reference test includes four-row inputs. Run that test
in separate processes with `ORT_HC_DECODE_ROWS=1` and `ORT_HC_DECODE_ROWS=2` to
exercise both dispatch choices.
