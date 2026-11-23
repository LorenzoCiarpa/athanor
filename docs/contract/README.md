# docs/contract

One file per model checkpoint (`<family>-<size>.md`, e.g. `qwen35-0.8b.md`) recording the exact op contract that the kernels must match:

- repository and revision of the checkpoint, reference implementation version, sha256 of the reference source files used;
- tensor names by layer type, with dtypes and shapes as reported by `athanor inspect`;
- every operation with its dtype boundaries, normalization details, RoPE parameters, gating, routing and state layout;
- the resolution of each **verify** item listed in `docs/MODELS.md`.

No kernel is written for an op whose contract entry is missing.
