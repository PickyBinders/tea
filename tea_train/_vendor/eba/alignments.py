import numpy as np
import numba as nb

MIN_FLOAT64 = np.finfo(np.float64).min

@nb.njit
def _make_dtw_matrix(
    score_matrix: np.ndarray,
    gap_open_penalty: float = 0.0,
    gap_extend_penalty: float = 0.0,
):
    """
    Make cost matrix using dynamic time warping
    Parameters
    ----------
    score_matrix
        matrix of scores between corresponding vectors of the two vector sets; shape = (n, m)
    gap_open_penalty
        penalty for opening a (series of) gap(s)
    gap_extend_penalty
        penalty for extending an existing series of gaps
    Returns
    -------
    accumulated cost matrix; shape = (n, m)
    """
    gap_open_penalty *= -1
    gap_extend_penalty *= -1
    n, m = score_matrix.shape
    matrix = np.zeros((n + 1, m + 1, 3), dtype=np.float64)
    matrix[:, 0, :] = MIN_FLOAT64
    matrix[0, :, :] = MIN_FLOAT64
    matrix[0, 0] = 0
    backtrack = np.zeros((n + 1, m + 1, 3), dtype=np.int64)
    for i in range(1, n + 1):
        matrix[i, 0, 0] = 0
        matrix[i, 0, 1] = 0
        matrix[i, 0, 2] = MIN_FLOAT64 - gap_open_penalty
        backtrack[i, 0] = 0

    for j in range(1, m + 1):
        matrix[0, j, 0] = MIN_FLOAT64 - gap_open_penalty
        matrix[0, j, 1] = 0
        matrix[0, j, 2] = 0
        backtrack[0, j] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores_lower = np.array(
                [
                    matrix[i - 1, j, 0] + gap_extend_penalty,
                    matrix[i - 1, j, 1] + gap_open_penalty,
                ]
            )
            index_lower = np.argmax(scores_lower)
            matrix[i, j, 0] = scores_lower[index_lower]
            backtrack[i, j, 0] = index_lower

            scores_upper = np.array(
                [
                    matrix[i, j - 1, 1] + gap_open_penalty,
                    matrix[i, j - 1, 2] + gap_extend_penalty,
                ]
            )
            index_upper = np.argmax(scores_upper)
            matrix[i, j, 2] = scores_upper[index_upper]
            backtrack[i, j, 2] = index_upper + 1

            scores = np.array(
                [
                    matrix[i, j, 0],
                    matrix[i - 1, j - 1, 1] + score_matrix[i - 1, j - 1],
                    matrix[i, j, 2],
                ]
            )
            index = np.argmax(scores)
            matrix[i, j, 1] = scores[index]
            backtrack[i, j, 1] = index
    return matrix, backtrack


@nb.njit
def _get_dtw_alignment(start_direction, backtrack: np.ndarray, n1, m1):
    """
    Finds optimal warping path from a backtrack matrix
    Parameters
    ----------
    start_direction
    backtrack
    n1
        length of first sequence
    m1
        length of second sequence
    Returns
    -------
    aligned_indices_1, aligned_indices_2
    """
    indices_1 = np.zeros(n1 + m1 + 1, dtype=np.int64)
    indices_2 = np.zeros(n1 + m1 + 1, dtype=np.int64)
    index = 0
    n, m = n1, m1
    direction = start_direction
    while not (n == 0 and m == 0):
        if m == 0:
            n -= 1
            indices_1[index] = n
            indices_2[index] = -1
            index += 1
        elif n == 0:
            m -= 1
            indices_1[index] = -1
            indices_2[index] = m
            index += 1
        else:
            if direction == 0:
                direction = backtrack[n, m, 0]
                n -= 1
                indices_1[index] = n
                indices_2[index] = -1
                index += 1
            elif direction == 1:
                direction = backtrack[n, m, 1]
                if direction == 1:
                    n -= 1
                    m -= 1
                    indices_1[index] = n
                    indices_2[index] = m
                    index += 1
            elif direction == 2:
                direction = backtrack[n, m, 2]
                m -= 1
                indices_1[index] = -1
                indices_2[index] = m
                index += 1
    return indices_1[:index][::-1], indices_2[:index][::-1]


@nb.njit
def dtw_align(
    score_matrix: np.ndarray,
    gap_open_penalty: float = 0.0,
    gap_extend_penalty: float = 0.0,
):
    """
    Align two objects using dynamic time warping
    Parameters
    ----------
    score_matrix
        (n x m) matrix of scores between all points of both objects
    gap_open_penalty
        penalty for opening a (series of) gap(s)
    gap_extend_penalty
        penalty for extending an existing series of gaps
    Returns
    -------
    aligned_indices_1, aligned_indices_2
    """
    matrix, backtrack = _make_dtw_matrix(
        score_matrix, gap_open_penalty, gap_extend_penalty
    )
    n = score_matrix.shape[0]
    m = score_matrix.shape[1]
    scores = np.array([matrix[n, m, 0], matrix[n, m, 1], matrix[n, m, 2]])
    index = np.argmax(scores)
    aln_1, aln_2 = _get_dtw_alignment(index, backtrack, n, m)
    return aln_1, aln_2, scores[index]
    #return scores[index]

###############################################
################LOCAL ALIGNMENT################
###############################################

@nb.njit
def unravel_index(idx, shape):
    row_idx = idx // shape[1]
    col_idx = idx % shape[1]
    return row_idx, col_idx

@nb.njit
def smith_waterman_matrix(score_matrix, gap=0.):
    rows = score_matrix.shape[0] + 1
    cols = score_matrix.shape[1] + 1
    matrix = np.zeros((rows, cols))

    for i in range(1, rows):
        for j in range(1, cols):
            # Calculate the score for each possible alignment
            diagonal_score = matrix[i - 1][j - 1] + score_matrix[i - 1, j - 1]
            left_score = matrix[i][j - 1] - gap
            up_score = matrix[i - 1][j] - gap
            # Take the maximum of the three possible scores
            matrix[i][j] = max(0, diagonal_score, left_score, up_score)
    return matrix

@nb.njit
def smith_waterman_traceback(score_matrix, matrix, gap=0.):
    i, j = unravel_index(np.argmax(matrix), matrix.shape)
    max_score = matrix[i, j]
    if max_score <= 0.:
        return None
    max_aln_length = i + j + 1
    align1 = np.zeros(max_aln_length, dtype=np.int64)
    align2 = np.zeros(max_aln_length, dtype=np.int64)
    index = 0
    first_1, first_2, last_1, last_2 = 0, 0, i, j
    while i > 0 and j > 0:
        score = matrix[i, j]
        diagonal_score = matrix[i - 1, j - 1]
        left_score = matrix[i, j - 1]
        up_score = matrix[i - 1, j]
        if score <= 0.:
            first_1, first_2 = i, j
            break
        elif score == diagonal_score + score_matrix[i - 1, j - 1]:
            i -= 1
            j -= 1
            align1[index] = i
            align2[index] = j
            index += 1
        elif score == left_score - gap:
            j -= 1
            align1[index] = -1
            align2[index] = j
            index += 1
        elif score == up_score - gap:
            i -= 1
            align1[index] = i
            align2[index] = -1
            index += 1
    if index > 2:
        return align1[:index][::-1], align2[:index][::-1], max_score, first_1, first_2, last_1, last_2
    else:
        return None

@nb.njit
def smith_waterman(score_matrix, gap=0., recurse=True):
    matrix = smith_waterman_matrix(score_matrix, gap)
    best_alignment = smith_waterman_traceback(score_matrix, matrix, gap)
    alignments = []
    if best_alignment is not None:
        aln1, aln2, score, first_1, first_2, last_1, last_2 = best_alignment
        alignments.append((aln1, aln2, score))
        if recurse:
            smith_waterman_recursive(score_matrix[:first_1], 0, 0, alignments, gap)
            smith_waterman_recursive(score_matrix[last_1:], last_1, 0, alignments, gap)
    return alignments


@nb.njit
def smith_waterman_recursive(score_matrix, add_1, add_2, alignments, gap=0.):
    matrix = smith_waterman_matrix(score_matrix, gap)
    best_alignment = smith_waterman_traceback(score_matrix, matrix, gap)
    if best_alignment is not None:
        aln1, aln2, score, first_1, first_2, last_1, last_2 = best_alignment
        # Preserve -1 gap markers when mapping recursive slices to full-sequence indices.
        alignments.append((np.where(aln1 >= 0, aln1 + add_1, -1), aln2, score))
        smith_waterman_recursive(score_matrix[:first_1], add_1, 0, alignments, gap)
        smith_waterman_recursive(score_matrix[last_1:], add_1 + last_1, 0, alignments, gap)
    return alignments
