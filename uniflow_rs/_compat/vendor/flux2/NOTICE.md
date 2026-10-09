This directory contains a source snapshot of the model and autoencoder
definitions from
[black-forest-labs/flux2](https://github.com/black-forest-labs/flux2).
The pinned upstream revision is
`50fe5162777813d869182b139e83b10743caef15`.
Those files are distributed under the Apache License 2.0; the license text is
available at the repository root in [LICENSE](../../../../LICENSE). The model forward
path has one local, clearly marked modification that adds optional
non-reentrant activation checkpointing; the autoencoder is unmodified. The
surrounding UniFlow-RS compatibility adapter is implemented outside this
directory.
