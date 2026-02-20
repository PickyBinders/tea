import numpy as np
import numba as nb
import subprocess
from pathlib import Path
from biotite import structure as bio_struct
from biotite.structure.io.pdb import PDBFile
from biotite.sequence.io.fasta import FastaFile
from biotite.structure import AffineTransformation

def run_usalign(
    pdb_file_1, pdb_file_2, matrices_folder, usalign_binary="USalign", overwrite=False
):
    q1, q2 = pdb_file_1.stem, pdb_file_2.stem
    fasta_file = matrices_folder / f"{q1}_{q2}.fasta"
    if not fasta_file.exists() or overwrite:
        with open(fasta_file, "w") as outfile:
            subprocess.run(
                [
                    usalign_binary,
                    str(pdb_file_1),
                    str(pdb_file_2),
                    "-mm",
                    "1",
                    "-ter",
                    "1",
                    "-m",
                    matrices_folder / f"{q1}_{q2}",
                    "-outfmt",
                    "1",
                ],
                stdout=outfile,
            )


def get_tm_score(filename):
    """
    gets TM score from a given fasta file returned by US-align

    Parameters
    ----------
    filename
        fasta file returned by US-align

    Returns
    -------
    min_tm_score, max_tm_score
    """
    min_tmscore = 2
    max_tmscore = 0
    sequences = {k:v for k, v in FastaFile.read(filename).items()}
    key = list(sequences.keys())[0]
    for key, _ in FastaFile.read_iter(filename):
        if key is None:
            return []
        tmscore = float(key.split("\t")[-1].split("=")[-1])
        if tmscore < min_tmscore:
            min_tmscore = tmscore
        if tmscore > max_tmscore:
            max_tmscore = tmscore
    return min_tmscore, max_tmscore


def alignment_to_numpy(alignment):
    aln_np = {}
    for n in alignment:
        aln_seq = []
        index = 0
        for a in alignment[n]:
            if a == "-":
                aln_seq.append(-1)
            else:
                aln_seq.append(index)
                index += 1
        aln_np[n] = np.array(aln_seq)
    return aln_np

def superpose_proteins(protein_1_file, protein_2_file, matrix_file):
    """
    Superposes two proteins using US-align-generated rotation, translation matrix

    Parameters
    ----------
    protein_1_file
        PDB file of protein_1
    protein_2_file
        PDB file of protein_2
    matrix_file
        US-align-generated rotation, translation matrix file
    Returns
    -------

    """
    matrix = np.zeros((3, 4))
    with open(matrix_file) as f:
        for i, line in enumerate(f):
            if 1 < i < 5:
                matrix[i - 2] = list(map(float, line.strip().split()[1:]))
    pdb_1 = PDBFile.read(protein_1_file).get_structure()[0]
    pdb_2 = PDBFile.read(protein_2_file).get_structure()[0]
    transformation = AffineTransformation(np.zeros(3), matrix[:, 1:], matrix[:, 0])
    pdb_1 = transformation.apply(pdb_1)
    return pdb_1, pdb_2


@nb.njit
def get_rmsd(coords_1: np.ndarray, coords_2: np.ndarray) -> float:
    """
    RMSD of paired coordinates = normalized square-root of sum of squares of euclidean distances
    """
    return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / coords_1.shape[0])


def get_info_json(id_1, id_2, pdb_folder, matrices_folder):
    fasta_file = matrices_folder / f"{id_1}_{id_2}.fasta"
    pdb_file_1 = pdb_folder / id_1[2:4] / f"{id_1}.ent"
    pdb_file_2 = pdb_folder / id_2[2:4] / f"{id_2}.ent"
    if not fasta_file.exists():
        run_usalign(pdb_file_1, pdb_file_2, matrices_folder)
    tm_score = get_tm_score(fasta_file)
    if tm_score[0] < 0.6:
        return None
    pdb_1, pdb_2 = superpose_proteins(pdb_file_1, pdb_file_2, matrices_folder / f"{id_1}_{id_2}")
    aln = {k: v for k, v in FastaFile.read(fasta_file).items()}
    aln = {k.split("\t")[0].split(":")[0].split("/")[-1].replace(".ent", ""): aln[k] for k in aln}
    max_length = min(len(aln[id_1]), len(aln[id_2]))
    aln = {k: v[:max_length] for k, v in aln.items()}
    coords_1 = pdb_1[pdb_1.atom_name == "CA"].coord
    coords_2 = pdb_2[pdb_2.atom_name == "CA"].coord
    aln_np = alignment_to_numpy(aln)
    rmsds = []
    for x in range(len(aln_np[id_1])):
        a1, a2 = aln_np[id_1][x], aln_np[id_2][x]
        if a1 == -1 or a2 == -1:
            rmsds.append(np.nan)
        else:
            rmsd = get_rmsd(coords_1[a1], coords_2[a2])
            rmsds.append(rmsd)
    assert len(rmsds) == len(aln_np[id_1]), f"Length of rmsds and aln_np[id_1] do not match for {id_1} {id_2}"
    assert len(rmsds) == len(aln_np[id_2]), f"Length of rmsds and aln_np[id_2] do not match for {id_1} {id_2}"
    return {
        "id_1": id_1,
        "id_2": id_2,
        "tm_score_min": tm_score[0],
        "tm_score_max": tm_score[1],
        "rmsds": rmsds,
        "aln": aln,
    }

def main():
    import json
    from sys import argv
    id_1, id_2, pdb_folder, matrices_folder = argv[1], argv[2], argv[3], argv[4]
    pdb_folder = Path(pdb_folder)
    matrices_folder = Path(matrices_folder)
    json_file = matrices_folder / f"{id_1}_{id_2}.json"
    if json_file.exists():
        print(f"Skipping {id_1} {id_2} because it already exists")
        return
    try:
        info = get_info_json(id_1, id_2, pdb_folder, matrices_folder)
        if info is None:
            print(f"Skipping {id_1} {id_2} because it has a TM score less than 0.6")
            return
        with open(json_file, "w") as f:
            json.dump(info, f)
    except Exception as e:
        print(f"Error {e} for {id_1} {id_2}")
        return

if __name__ == "__main__":
    main()