# GPU-4A background-run record

GPU-3C's independent supervisor/worker, duplicate-run lock, logs, success/failure state handling, timeout behavior, and process cleanup remain covered by regression tests. GPU-4A prewarm measured 0.155 seconds/step and supported an approximately 75-second estimate for 300 Full SFT steps; actual training took 48.9 seconds. Baseline and SFT generation evaluations took 43.1 and 13.7 seconds.

No measured GPU-4A task was expected to exceed five minutes, so no GPU-4A background job was required. Status: `NOT_REQUIRED_FOR_GPU4A_MEASURED_TASKS`.
