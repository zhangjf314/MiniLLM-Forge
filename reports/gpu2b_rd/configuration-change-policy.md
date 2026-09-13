# Configuration-change policy

Branch H authorizes no scientific or training configuration change. Historical GPU-2B
must never be made to pass by lowering its 1,536 MiB gate or silently changing sequence
length, corpus, token budget, accumulation, optimizer, precision, adapter rank, or required
modules.

Branch C remains a separate fallback design route and is not selected. If adequate
Branch-H hardware is prospectively established as unavailable, a new
`STAGE_GPU_2B_D2` zero-training design stage may consider bounded changes to activation or
gradient checkpointing, optimizer implementation, CPU/offload strategy, sequence-length
policy, or packing implementation.

Any D2 lever must be justified before numerical results and applied identically to Base
and CPT initializations within each adaptation family. Multiple dimensions must not be
changed merely until a run fits. D2 may not modify or relabel GPU-2B-D or the blocked
GPU-2B evidence.
