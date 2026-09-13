# Hardware authority for future H1

Before any `STAGE_GPU_2B_H1` numerical launch, seal one environment inventory containing:

- GPU model and physical VRAM;
- NVIDIA driver, CUDA, PyTorch, bitsandbytes, Transformers, and PEFT versions;
- operating system, execution mode, and display/WDDM status where relevant;
- CPU model, installed RAM, and available storage.

Preferred capacity is at least 12 GiB physical VRAM. Qualification still passes only when
each family records at least 1,536 MiB minimum physical headroom, 64/64 optimizer updates,
and zero NaN, Inf, and OOM events.

Physical-device monitoring is authoritative. H1 must record both independent physical GPU
usage/headroom and framework allocated/reserved counters. Allocator counters alone are
insufficient on a platform where they do not represent physical occupancy.

All 18 future formal runs should use the same frozen hardware class and environment.
Base-init and CPT-init arms must not be split across unequal hardware and then treated as
paired runtime evidence. Any explicitly preregistered scheduler-equivalence policy must be
frozen before H1 Q0; none is introduced by RD.
