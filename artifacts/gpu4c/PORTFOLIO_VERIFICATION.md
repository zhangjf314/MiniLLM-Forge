# Portfolio Verification

Quick verification: **PASS** (29/29 checks, 0.81 s).

No training, generation campaign, network download, performance benchmark, or API call was executed.

| Check | Status | Detail |
|---|---|---|
| hash:artifacts/training/minillm_formal_result.json | PASS | expected=6c27dbf40736919a21c8bd69f68f0f4e3e91ea0b46cfe58f397e101a40874f6a; actual=6c27dbf40736919a21c8bd69f68f0f4e3e91ea0b46cfe58f397e101a40874f6a |
| hash:artifacts/training/minillm_formal_model.json | PASS | expected=de4afdc0535a6e2b532a5ec156a1798a57bf1516616d3e7daa04cda9d37c5bbc; actual=de4afdc0535a6e2b532a5ec156a1798a57bf1516616d3e7daa04cda9d37c5bbc |
| hash:artifacts/training/minillm_gpu_resume_validation.json | PASS | expected=eb5fbb8be5220692f8203ddc5a68f21c06ac06d8855a8812fc7f3d22ecf62e15; actual=eb5fbb8be5220692f8203ddc5a68f21c06ac06d8855a8812fc7f3d22ecf62e15 |
| hash:artifacts/tokenizers/minillm-tokenizer.json | PASS | expected=e9578dfd6d2a0f4c3137d7cc223a7b8a782afba8190ae1ad105e1b8ef37e9a22; actual=e9578dfd6d2a0f4c3137d7cc223a7b8a782afba8190ae1ad105e1b8ef37e9a22 |
| hash:artifacts/data_manifests/minillm_pretrain_formal.json | PASS | expected=44989a2a0fc8b41e2ecea3f5f5a5e312315dec4333fc8f4106844cfff7256103; actual=44989a2a0fc8b41e2ecea3f5f5a5e312315dec4333fc8f4106844cfff7256103 |
| hash:artifacts/gpu2d_formal/final_result.json | PASS | expected=d2c1397a1541bc400054287915639393dee9d68b4a0f73fb4156d78a3074af93; actual=d2c1397a1541bc400054287915639393dee9d68b4a0f73fb4156d78a3074af93 |
| hash:artifacts/gpu3a/gpu3a_result.json | PASS | expected=13736c692dabfc61be35609104357995786e2db6ea038b5db417f854d8ce6dc6; actual=13736c692dabfc61be35609104357995786e2db6ea038b5db417f854d8ce6dc6 |
| hash:artifacts/gpu3b/gpu3b_result.json | PASS | expected=d9a0366977114305944f11041ddad0cc93222131764d6370d97e60de89ce5ac2; actual=d9a0366977114305944f11041ddad0cc93222131764d6370d97e60de89ce5ac2 |
| hash:artifacts/gpu3c/gpu3c_result.json | PASS | expected=5dc81f0591262ab7f8728bc53e1345034dab629b92ea28d8bced37f7522023a8; actual=5dc81f0591262ab7f8728bc53e1345034dab629b92ea28d8bced37f7522023a8 |
| hash:artifacts/gpu4a/gpu4a_result.json | PASS | expected=c9745b15005e5a6a7b404ffb2383b74ee155091910c2f7b894b2e191fc7fa285; actual=c9745b15005e5a6a7b404ffb2383b74ee155091910c2f7b894b2e191fc7fa285 |
| hash:artifacts/gpu4a/formal_sft_result.json | PASS | expected=acad8399bc56e7a8c1230198f07ababfbe7936a6a276170131056987f1f3f9b3; actual=acad8399bc56e7a8c1230198f07ababfbe7936a6a276170131056987f1f3f9b3 |
| hash:artifacts/gpu4a/sft_evaluation.json | PASS | expected=4cf215e25163271988a0ed39bc20f3cdfbdd6c1091cdf041e282c76fc9b890ca; actual=4cf215e25163271988a0ed39bc20f3cdfbdd6c1091cdf041e282c76fc9b890ca |
| hash:artifacts/gpu4b/gpu4b_result.json | PASS | expected=79d72186870c08ad84f68d9bdec572088cab8ab2be9d649324fc9c73be5fb7c0; actual=79d72186870c08ad84f68d9bdec572088cab8ab2be9d649324fc9c73be5fb7c0 |
| hash:artifacts/gpu4b_r1/gpu4b_r1_result.json | PASS | expected=69b1cc95f40640a089784a513900c4073dd6e87564c9fe922788552f1edd947d; actual=69b1cc95f40640a089784a513900c4073dd6e87564c9fe922788552f1edd947d |
| hash:runs/E01-minillm-formal/best.pt | PASS | expected=ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc; actual=ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc |
| hash:runs/gpu4a/formal-full-sft-s42-v1/checkpoints/step-0300.pt | PASS | expected=9c33b1c5a781c4ab1d9832a0aad1349516fafb011d24ff6c67be3b1d3885b96e; actual=9c33b1c5a781c4ab1d9832a0aad1349516fafb011d24ff6c67be3b1d3885b96e |
| manifest_schema | PASS | required final evidence fields |
| claim_schema | PASS | claims=23 |
| validated_claim_evidence | PASS | every VALIDATED claim has a repository artifact |
| experiment_schema | PASS | experiments=23 |
| tokenizer_load | PASS | vocab_size=24000 |
| pretrained_checkpoint_metadata | PASS | global_step=3052; model_state=True |
| native_sft_checkpoint_metadata | PASS | global_step=300; model_state=True |
| t1_subset_denominator | PASS | fixed independent subset=64/64; not full test |
| gpu3b_evaluator_boundary | PASS | extraction/stopping improvement does not change model |
| gpu3c_negative_preserved | PASS | GPU3C_CONTROLLED_NEGATIVE_RESULT |
| gpu4b_partial_preserved | PASS | GPU4B_PARTIAL_VALIDATION |
| gpu4b_original_gate_preserved | PASS | BF16=false; production unchanged |
| quick_has_no_training | PASS | quick performs read-only verification except its own report |
