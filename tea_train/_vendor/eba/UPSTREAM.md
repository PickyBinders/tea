# EBA numerical kernels

Source: https://git.scicore.unibas.ch/schwede/EBA

Revision: `7fc3b4a460520eb178a6d570d78dfb745a627903`

`alignments.py` preserves `-1` gap markers when applying recursive row
offsets to residue indices. TEA uses `alignments.smith_waterman(..., recurse=False)`
for the best local alignment and `score_matrices.compute_similarity_matrix`
for normalized embedding similarity. Both kernels are distributed under
Apache-2.0 with the upstream LICENSE and NOTICE.
