# Source provenance

This anonymous release is intentionally restricted to the final paper
backbone and contains no legacy FLUX.1 execution path.

## Upstream Apache-2.0 source

The files below originate from the official Black Forest Labs FLUX.2
repository and retain Apache-2.0 attribution:

```text
uniflow_rs/_compat/vendor/flux2/model.py
uniflow_rs/_compat/vendor/flux2/autoencoder.py
```

Upstream repository and pinned revision:

```text
https://github.com/black-forest-labs/flux2
50fe5162777813d869182b139e83b10743caef15
```

The model file contains one marked modification: optional non-reentrant
activation checkpointing for memory-efficient LoRA training. The autoencoder
implementation is unchanged apart from its provenance header.

The corresponding FLUX.2 Klein Base 4B model card declares Apache-2.0. Model
weights are not redistributed by this repository.

## UniFlow-RS source

All code outside the vendored directory implements the paper method or its
benchmark interface: the three-stream adapter, shared LoRA, source-state
construction, structured visual targets, TMAS, data preparation, metrics,
checkpoint validation, and prediction export.

The released checkpoint uses the sanitized identifiers
`uniflow_rs_lora_v2` and
`uniflow-rs-flux2-klein-base-4b-multitask`. Only metadata was renamed; LoRA
tensors and parameter keys are unchanged.

## Excluded source

The package excludes all unused legacy backbone implementations, text
conditioners, model loaders, and sampling helpers. It does not require another
research repository at runtime.

## External assets

Qwen weights and all benchmark datasets are downloaded separately. They are
not covered by this repository's Apache-2.0 grant; users must follow their
respective upstream terms.
