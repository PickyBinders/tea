"""Sequence-only local EBA supervision using the upstream EBA local mode."""

from collections import defaultdict
import math

import numpy as np

from tea_train.io import stable_hash
from tea_train._vendor.eba.alignments import smith_waterman


def affine_sweep_scores(similarity, gaps):
    """Reuse one Parasail PSSM for an EBA gap sweep, with Biotite fallback.

    Unique target-position bytes encode the positional similarities. Use Biotite
    when Parasail returns adjacent opposing gaps. Return the optimal fixed-point
    scores in the original units.
    """
    import ctypes
    import parasail
    from parasail import bindings_v2 as native
    from biotite.sequence import Alphabet, GeneralSequence
    from biotite.sequence.align import SubstitutionMatrix, align_optimal

    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or min(similarity.shape) < 2 or not np.isfinite(similarity).all():
        raise ValueError("Finite positional matrix with both lengths >= 2 required")
    if not gaps or any(not math.isfinite(go + ge) or go <= 0 or ge < 0 for go, ge in gaps):
        raise ValueError("Invalid affine gap grid")
    if similarity.shape[1] > similarity.shape[0]:
        similarity = similarity.T
    symbols = bytes(value for value in range(1, 256) if value != ord('*'))
    bound = min(np.maximum(similarity, 0).max(0).sum(), np.maximum(similarity, 0).max(1).sum())
    safe = int((np.iinfo(np.int32).max / 4) /
               (bound + max(similarity.shape) + max(go + ge for go, ge in gaps) + np.abs(similarity).max()))
    scale = min(10000, (safe // 2000) * 1000)
    if scale < 1000:
        raise ValueError("Alignment exceeds the validated fixed-point score range")
    fixed = np.ascontiguousarray(np.rint(similarity * scale), dtype=np.int32)
    if fixed.max() <= 0:
        return [0.0] * len(gaps)
    eligible = similarity.shape[1] <= len(symbols)
    if eligible:
        # The pinned Python wrapper omits this documented native PSSM constructor.
        create = native._lib.parasail_matrix_pssm_create_case_sensitive
        create.argtypes = [ctypes.c_char_p, native.c_int_p, ctypes.c_int]
        create.restype = native.c_matrix_p
        target = symbols[:similarity.shape[1]]
        pointer = create(target, fixed.ctypes.data_as(native.c_int_p), similarity.shape[0])
        if not pointer:
            raise RuntimeError("Parasail PSSM creation failed")
        matrix = parasail.Matrix(pointer)
        profile = parasail.profile_create_32(b'A' * similarity.shape[0], matrix)
    fallback = None
    scores = []
    for opening, extension in gaps:
        use_native = eligible and opening >= extension
        if use_native:
            result = parasail.sw_trace_striped_profile_32(profile, target, round(opening * scale), round(extension * scale))
            # Positive cells imply a positive local optimum. A negative
            # result is a native PSSM sentinel and triggers the fallback.
            use_native = not result.saturated and result.score > 0
            if use_native:
                cigar = result.cigar
                operations = cigar.seq & 15
                opposite = np.any(((operations[:-1] == 1) & (operations[1:] == 2)) |
                                  ((operations[:-1] == 2) & (operations[1:] == 1)))
                use_native = not opposite and cigar.len > 0
        if use_native:
            scores.append(float(result.score) / scale)
        else:
            if fallback is None:
                sequences = [GeneralSequence(Alphabet(range(n)), np.arange(n)) for n in similarity.shape]
                fallback = SubstitutionMatrix(*(seq.get_alphabet() for seq in sequences), fixed)
            result = align_optimal(*sequences, fallback,
                                   gap_penalty=(-round(opening * scale), -round(extension * scale)),
                                   local=True, max_number=1)
            scores.append(float(result[0].score) / scale if result else 0.0)
    return scores


def local_from_similarity(similarity, gap=0.0, *, gap_open=None, gap_extend=None, min_paired_residues=3):
    """Return the best local EBA traceback."""
    similarity = np.asarray(similarity, dtype=np.float64)
    if min_paired_residues < 1:
        raise ValueError("At least one paired residue is required")
    if similarity.ndim != 2 or min(similarity.shape) < 2 or not np.isfinite(similarity).all():
        raise ValueError("Local EBA requires a finite 2-D similarity matrix with both lengths >= 2")
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("Local EBA uses a nonnegative linear gap penalty")
    affine = gap_open is not None or gap_extend is not None
    extra = {}
    if affine:
        if (gap != 0 or gap_open is None or gap_extend is None or
                not math.isfinite(gap_open + gap_extend) or gap_open <= 0 or gap_extend < 0):
            raise ValueError("Affine gaps require positive opening, nonnegative extension and no linear gap")
        from biotite.sequence import Alphabet, GeneralSequence
        from biotite.sequence.align import SubstitutionMatrix, align_optimal
        # EBA supplies the positional scores; Biotite supplies local Gotoh DP.
        # Integer scaling preserves fractional EBA scores, with int32 headroom.
        bound = min(np.maximum(similarity, 0).max(0).sum(), np.maximum(similarity, 0).max(1).sum())
        safe = int((np.iinfo(np.int32).max / 4) /
                   (bound + max(similarity.shape) + gap_open + gap_extend + np.abs(similarity).max()))
        scale = min(10000, (safe // 2000) * 1000)
        if scale < 1000:
            raise ValueError("Alignment exceeds the validated fixed-point score range")
        sequences = [GeneralSequence(Alphabet(range(n)), np.arange(n)) for n in similarity.shape]
        matrix = SubstitutionMatrix(*(seq.get_alphabet() for seq in sequences),
                                    np.rint(similarity * scale).astype(np.int32))
        alignments = align_optimal(*sequences, matrix,
                                  gap_penalty=(-round(gap_open * scale), -round(gap_extend * scale)),
                                  local=True, max_number=1)
        if not alignments:
            return None
        trace = alignments[0].trace
        a, b = trace.T
        paired = (a >= 0) & (b >= 0)
        runs = sum(int(np.count_nonzero(np.diff(np.r_[False, trace[:, col] < 0, False].astype(int)) == 1))
                   for col in (0, 1))
        cost = gap_open * runs + gap_extend * (int((~paired).sum()) - runs)
        score = float(similarity[a[paired], b[paired]].sum() - cost)
        if abs(score - float(alignments[0].score) / scale) > paired.sum() * .5 / scale + 1e-6:
            raise ValueError("Affine traceback does not reconstruct its fixed-point score")
        extra = {"alignment_backend": "biotite-local-gotoh", "gap_open": gap_open,
                 "gap_extend": gap_extend, "gap_runs": runs, "fixed_point_scale": scale,
                 "paired_substitution_scores": similarity[a[paired], b[paired]].tolist()}
    else:
        alignments = smith_waterman(similarity, gap=float(gap), recurse=False)
        if not alignments:
            return None
        a, b, score = alignments[0]
        cost = gap * ((a < 0) | (b < 0)).sum()
    if a.shape != b.shape or ((a < 0) & (b < 0)).any():
        raise ValueError("Invalid EBA traceback")
    for indices, length in zip((a, b), similarity.shape):
        valid = indices[indices >= 0]
        if (indices < -1).any() or (valid >= length).any() or (np.diff(valid) <= 0).any():
            raise ValueError("EBA traceback outside full-chain coordinates")
    paired = (a >= 0) & (b >= 0)
    if paired.sum() < min_paired_residues:
        return None
    reconstructed = similarity[a[paired], b[paired]].sum() - cost
    if not np.isclose(reconstructed, score, rtol=1e-8, atol=1e-8):
        raise ValueError("EBA traceback does not reconstruct local score")
    return {"mode": "eba-local-best", "raw_score": float(score),
            "mean_paired_score": float(similarity[a[paired], b[paired]].mean()),
            "paired_residues": int(paired.sum()), "gap_columns": int((~paired).sum()),
            "gap_penalty": None if affine else gap, "length_a": similarity.shape[0], "length_b": similarity.shape[1],
            "index_a": a.tolist(), "index_b": b.tolist(),
            "coverage_a": float(paired.sum() / similarity.shape[0]),
            "coverage_b": float(paired.sum() / similarity.shape[1]), **extra}


def local_from_embeddings(embedding_a, embedding_b, distance_scale=1.0, gap=0.0, *, gap_open=None, gap_extend=None):
    from tea_train._vendor.eba.score_matrices import compute_similarity_matrix
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("Positive finite EBA distance scale required")
    similarity = compute_similarity_matrix(embedding_a.float(), embedding_b.float(), l=distance_scale, p=2)
    return local_from_similarity(similarity.detach().cpu().numpy(), gap, gap_open=gap_open, gap_extend=gap_extend)


def family_candidates(mapping, identifiers):
    """Yield unique pairs of full chains that share a family."""
    identifiers = sorted(set(identifiers))
    members = defaultdict(set)
    for key in identifiers:
        for family in mapping["chains"][key]["families"]:
            members[family].add(key)
    for left in identifiers:
        families = set(mapping["chains"][left]["families"])
        targets = set().union(*(members[family] for family in families))
        for right in sorted(targets):
            if right <= left:
                continue
            shared = sorted(families & set(mapping["chains"][right]["families"]))
            yield left, right, shared


def aligned_columns(alignment, chain_a, chain_b, mapping):
    """Yield all paired residues in the local EBA alignment for matrix fitting."""
    shared = sorted(set(mapping["chains"][chain_a]["families"]) &
                    set(mapping["chains"][chain_b]["families"]))
    if not shared or chain_a == chain_b or alignment["mode"] != "eba-local-best":
        raise ValueError("Expected a same-family, non-self local EBA alignment")
    alignment_id = stable_hash({"chain_a": chain_a, "chain_b": chain_b, "alignment": alignment})
    scores = alignment.get("paired_substitution_scores")
    if scores is not None and (len(scores) != alignment["paired_residues"] or not np.isfinite(scores).all()):
        raise ValueError("Invalid aligned substitution scores")
    position = 0
    for a, b in zip(alignment["index_a"], alignment["index_b"]):
        if a < 0 or b < 0:
            continue
        yield {"kind": "eba_aligned", "chain_a": chain_a, "chain_b": chain_b,
               "index_a": a, "index_b": b, "alignment_id": alignment_id,
               "shared_families": shared,
               **({"substitution_score": scores[position]} if scores is not None else {})}
        position += 1


def family_regions(mapping, chain_id, index):
    return {mapping["domains"][domain]["family"]
            for domain in mapping["chains"][chain_id]["domains"]
            for region in mapping["domains"][domain]["slices"]
            if region["chain_id"] == chain_id and region["start"] <= index < region["stop"]}


def training_positive(column, mapping):
    """Check whether paired residues lie in spans that share a family."""
    return bool(family_regions(mapping, column["chain_a"], column["index_a"]) &
                family_regions(mapping, column["chain_b"], column["index_b"]))
