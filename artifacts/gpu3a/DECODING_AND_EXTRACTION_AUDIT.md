# Decoding and Extraction Audit

The frozen evaluator uses greedy decoding (`do_sample=false`, one beam), 512 new tokens, stop IDs
151643/151645 and pad ID 151643. It trims through the first stop token and stores token IDs. The
stored text hides special tokens, so termination analysis must use IDs.

Verified results: 8398/8400 records ended at the length
limit; 2 ended on a recognized stop token. There is no evidence that EOS
was generated but ignored. The two EOS-ended records emitted token 151643.

The legacy extractor selects the last balanced `\boxed{...}`, otherwise the last answer-pattern
match, otherwise the final non-empty line. All 36 empty legacy
answers are length-limited. Diagnostic parsing shows an earlier non-empty answer in these records,
followed by continued/repeated generation and a final empty answer marker. Therefore the historical
36 are confirmed extraction-rule mismatches compounded by generation truncation; historical scores
remain untouched.
