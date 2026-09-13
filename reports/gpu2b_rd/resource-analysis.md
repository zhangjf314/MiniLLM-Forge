# Resource analysis

The observed physical-memory requirement and frozen safety margin imply:

```text
7,776 MiB observed use
+ 1,536 MiB required headroom
= 9,312 MiB lower resource bound
```

This is a lower bound derived from one observed qualification, not a guarantee that a
nominal 10 GB device will pass. Driver, runtime, allocator, display, WDDM, and sequence-
length variation can consume the apparent margin.

The selected Branch-H target is therefore at least 12 GiB of physical VRAM. The additional
capacity is an engineering recommendation and does not replace or raise the scientific
experiment's unchanged 1,536 MiB measured-headroom gate.

The blocked environment was:

- NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB physical VRAM;
- NVIDIA driver 577.02, CUDA runtime reported by PyTorch 12.8;
- PyTorch 2.11.0+cu128, bitsandbytes 0.50.2;
- Transformers 5.16.1, PEFT 0.20.0;
- Windows 11 build 26100, WDDM execution, display inactive for the device;
- Intel Core i7-14650HX, 16,763,232,256 bytes installed RAM;
- approximately 386.58 GiB free on the project volume at RD freeze time.

These facts describe the blocked platform. They are not automatically the H1 environment.
