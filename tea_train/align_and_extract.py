"""Create the structural alignments used by the original TEA training recipe."""

import json
import subprocess
from pathlib import Path

import numpy as np
from biotite.sequence.align import Alignment
from biotite.sequence.io.fasta import FastaFile
from biotite.structure import AffineTransformation
from biotite.structure.io.pdb import PDBFile
from lightning.pytorch.cli import LightningArgumentParser


def run_usalign(
    pdb_file_1,
    pdb_file_2,
    output_folder,
    usalign_binary="USalign",
    overwrite=False,
):
    """Run US-align and retain its alignment and transformation matrix."""
    pair = f"{pdb_file_1.stem}_{pdb_file_2.stem}"
    alignment_file = output_folder / f"{pair}.fasta"
    matrix_file = output_folder / pair
    if alignment_file.exists() and matrix_file.exists() and not overwrite:
        return alignment_file, matrix_file
    with alignment_file.open("w") as output:
        subprocess.run(
            [
                str(usalign_binary),
                str(pdb_file_1),
                str(pdb_file_2),
                "-mm",
                "1",
                "-ter",
                "1",
                "-m",
                str(matrix_file),
                "-outfmt",
                "1",
            ],
            stdout=output,
            check=True,
        )
    return alignment_file, matrix_file


def read_usalign_alignment(path):
    records = list(FastaFile.read(path).items())
    ids = [header.split("\t")[0].split(":")[0].split("/")[-1].replace(".ent", "")
           for header, _ in records]
    sequences = [sequence for _, sequence in records]
    tm_scores = [float(header.split("\t")[-1].split("=")[-1]) for header, _ in records]
    length = min(map(len, sequences))
    sequences = [sequence[:length] for sequence in sequences]
    trace = Alignment.trace_from_strings(sequences)
    return ids, sequences, trace, tm_scores


def get_tm_score(filename):
    """Return the minimum and maximum TM-scores from US-align output."""
    *_, tm_scores = read_usalign_alignment(filename)
    return min(tm_scores), max(tm_scores)


def alignment_to_numpy(alignment):
    """Return the historical per-column residue-index representation."""
    trace = Alignment.trace_from_strings(list(alignment.values()))
    return {
        identifier: trace[:, index]
        for index, identifier in enumerate(alignment)
    }


def transform_structures(protein_1, protein_2, matrix_file):
    matrix = np.loadtxt(matrix_file, skiprows=2, max_rows=3)[:, 1:]
    structure_1 = PDBFile.read(protein_1).get_structure(model=1)
    structure_2 = PDBFile.read(protein_2).get_structure(model=1)
    transform = AffineTransformation(np.zeros(3), matrix[:, 1:], matrix[:, 0])
    return transform.apply(structure_1), structure_2


superpose_proteins = transform_structures


def extract_pair(id_1, id_2, pdb_folder, output_folder, usalign="USalign"):
    """Return the original TEA structural-alignment JSON record."""
    pdb_file_1 = pdb_folder / id_1[2:4] / f"{id_1}.ent"
    pdb_file_2 = pdb_folder / id_2[2:4] / f"{id_2}.ent"
    alignment_file, matrix_file = run_usalign(
        pdb_file_1, pdb_file_2, output_folder, usalign
    )
    ids, sequences, trace, tm_scores = read_usalign_alignment(alignment_file)
    if set(ids) != {id_1, id_2}:
        raise ValueError(f"US-align returned unexpected identifiers: {ids}")

    structure_1, structure_2 = transform_structures(
        pdb_file_1, pdb_file_2, matrix_file
    )
    coordinates = {
        id_1: structure_1[structure_1.atom_name == "CA"].coord,
        id_2: structure_2[structure_2.atom_name == "CA"].coord,
    }
    order = {identifier: i for i, identifier in enumerate(ids)}
    rmsds = []
    for column in trace:
        index_1, index_2 = column[order[id_1]], column[order[id_2]]
        if index_1 == -1 or index_2 == -1:
            rmsds.append(None)
        else:
            rmsds.append(float(np.linalg.norm(
                coordinates[id_1][index_1] - coordinates[id_2][index_2]
            )))
    return {
        "id_1": id_1,
        "id_2": id_2,
        "tm_score_min": min(tm_scores),
        "tm_score_max": max(tm_scores),
        "rmsds": rmsds,
        "aln": {identifier: sequences[order[identifier]] for identifier in ids},
    }


def get_info_json(id_1, id_2, pdb_folder, matrices_folder):
    """Retain the original structural-preparation library entry point."""
    record = extract_pair(id_1, id_2, pdb_folder, matrices_folder)
    return record if record["tm_score_min"] >= 0.6 else None


def main():
    parser = LightningArgumentParser(description=__doc__)
    parser.add_argument("id_1")
    parser.add_argument("id_2")
    parser.add_argument("pdb_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--usalign", type=Path, default=Path("USalign"))
    parser.add_argument("--tm-score-threshold", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)
    output = args.output_folder / f"{args.id_1}_{args.id_2}.json"
    if output.exists() and not args.overwrite:
        print(f"Skipping existing {output}")
        return
    record = extract_pair(
        args.id_1, args.id_2, args.pdb_folder, args.output_folder, args.usalign
    )
    if record["tm_score_min"] < args.tm_score_threshold:
        print(f"Skipping {args.id_1} {args.id_2}: TM-score below {args.tm_score_threshold}")
        return
    with output.open("w") as handle:
        json.dump(record, handle)


if __name__ == "__main__":
    main()
