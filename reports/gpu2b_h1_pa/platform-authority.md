# Platform authority

`H1_PLATFORM_AVAILABLE` requires realistic, authorized execution access. Hardware that
merely exists commercially, or cloud capacity that could theoretically be rented, does
not satisfy this audit.

The inventory contains one accessible candidate, `LOCAL-RTX5060L-8GB`. Its software and
hardware identity was inspected without training. It is classified `H1_ELIGIBLE = NO`
because this exact platform is the historical approximately 8 GB environment whose frozen
Full-SFT Q64 run recorded only 35 MiB minimum physical headroom.

The empirical capacity lower bound is 7,776 + 1,536 = 9,312 MiB. The preferred H1 target
remains at least 12 GiB physical VRAM, but this is an engineering margin rather than the
formal gate or a guarantee of qualification.

No authorized remote server, institutional GPU, or cloud GPU is recorded. Codex exposes
only the local execution host for the project's saved environments. Therefore there is no
eligible environment to freeze as `selected_environment`.
