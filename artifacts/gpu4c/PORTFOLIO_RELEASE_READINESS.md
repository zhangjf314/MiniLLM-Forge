# Portfolio Release Readiness

Candidate version: `v2.0.0-portfolio` (not created by GPU-4C).

| Gate | Status | Evidence |
|---|---|---|
| Key checkpoints and hashes | PASS | final evidence manifest + quick verification |
| Key artifacts and schemas | PASS | claim/experiment registries |
| README and final report truthfulness | PASS_WITH_LIMITATIONS | documentation truth audit |
| Historical experiment artifacts | PASS | frozen SHA256 set |
| Quick verification | PASS | 29/29 bounded checks |
| Full regression | PASS | 218 passed; 6 expected forced-Flash warnings |
| No stale experimental path presented as production | PASS | Manual + Eager remains formal |

Release posture: `READY_WITH_DOCUMENTED_LIMITATIONS`. Subset evaluation, unavailable Flash, missing Triton compile path, unvalidated Qwen Full SFT effectiveness, and absent distributed training are explicitly visible. Quick verification and full regression passed. A `v2.0.0-portfolio` tag is recommended after the final commit is reviewed and the worktree is confirmed clean; GPU-4C does not create the tag.
