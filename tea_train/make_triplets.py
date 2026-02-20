import numpy as np
import numba as nb
import pandas as pnd
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from dataclasses import dataclass
from tea.train.align_and_extract import alignment_to_numpy
import json
import argparse

@dataclass
class ResidueTriplet:
    protein_1: str
    protein_2: str
    index_1: int
    index_2_positive: int
    index_2_negative: int
    shift: int
    aa1: str
    aa2_positive: str
    aa2_negative: str
    rmsd: float

    def __str__(self):
        return "\t".join(
            [
                self.protein_1.split("/")[0],
                self.protein_2.split("/")[0],
                self.protein_1.split("/")[1],
                self.protein_2.split("/")[1],
                str(self.index_1),
                str(self.index_2_positive),
                str(self.index_2_negative),
                str(self.shift),
                self.aa1,
                self.aa2_positive,
                self.aa2_negative,
                f"{self.rmsd:.3f}",
            ]
        )

def sample_from_protein_pair_scop(
    data: dict,
    negative_start=1,
    negative_end=5,
    tm_score_threshold=0.6,
    rmsd_threshold=5,
):
    """
    Creates ResidueTriplet objects from two aligned proteins

    Parameters
    ----------
    data
        dict with 'id_1', 'id_2', 'tm_score_min', 'aln', 'rmsds'
    negative_start
        start of negative pair
    negative_end
        end of negative pair
    tm_score_threshold
        TM-score threshold
    rmsd_threshold
        RMSD threshold
    
    Returns
    -------
    list of ResidueTriplet objects
    """

    
    if data['tm_score_min'] < tm_score_threshold:
        return
    # remove columns with gap in both sequences
    aln_1 = data['aln'][data['id_1']]
    aln_2 = data['aln'][data['id_2']]
    keep_indices = [
        i
        for i in range(len(aln_1))
        if aln_1[i] != "-" or aln_2[i] != "-"
    ]
    aln_pair = {
        data['id_1']: "".join([aln_1[i] for i in keep_indices]),
        data['id_2']: "".join([aln_2[i] for i in keep_indices]),
    }
    aln_np = alignment_to_numpy(aln_pair)

    for x in range(len(aln_np[data['id_1']])):
        if data['rmsds'][x] is None or np.isnan(data['rmsds'][x]) or data['rmsds'][x] > rmsd_threshold:
            continue
        a1, a2_positive = aln_np[data['id_1']][x], aln_np[data['id_2']][x]
        if a1 == -1 or a2_positive == -1:
            continue
        negative_indices = []
        # select [x-negative_end:x-negative_start] and [x+negative_start:x+negative_end] as negative pairs
        for i in range(x - negative_end, x - negative_start + 1):
            if i >= 0 and i < len(aln_np[data['id_2']]) and aln_np[data['id_2']][i] != -1:
                negative_indices.append((i, aln_np[data['id_2']][i]))
        for i in range(x + negative_start, x + negative_end + 1):
            if i >= 0 and i < len(aln_np[data['id_2']]) and aln_np[data['id_2']][i] != -1:
                negative_indices.append((i, aln_np[data['id_2']][i]))

        for i, a2_negative in negative_indices:
            yield ResidueTriplet(
                protein_1=f"{data['id_1']}/1-{len(aln_1.replace('-', ''))}",
                protein_2=f"{data['id_2']}/1-{len(aln_2.replace('-', ''))}",
                index_1=a1,
                index_2_positive=a2_positive,
                index_2_negative=a2_negative,
                rmsd=data['rmsds'][x],
                shift=i - x,
                aa1=aln_pair[data['id_1']][x],
                aa2_positive=aln_pair[data['id_2']][x],
                aa2_negative=aln_pair[data['id_2']][i],
            )


def make_training_data_triplet_scop(json_file,
        output_file,
        negative_start=1,
        negative_end=10,
        overwrite=False,
        tm_score_threshold=0.6,
        rmsd_threshold=5):
    with open(json_file) as f:
        data = json.load(f)
    num = 0
    if not output_file.exists() or overwrite:
        with open(output_file, "w") as info_file:
            header = [
                "protein_1",
                "protein_2",
                "range_1",
                "range_2",
                "index_1",
                "index_2_positive",
                "index_2_negative",
                "shift",
                "aa1",
                "aa2_positive",
                "aa2_negative",
                "rmsd",
            ]
            info_file.write("\t".join(header) + "\n")

            for residue_pair in sample_from_protein_pair_scop(
                data,
                negative_start=negative_start,
                negative_end=negative_end,
                tm_score_threshold=tm_score_threshold,
                rmsd_threshold=rmsd_threshold,
            ):
                if residue_pair is None:
                    continue
                num += 1
                info_file.write(
                    f"{str(residue_pair)}\n"
                )
    return num

def main():
    parser = argparse.ArgumentParser(description='Process SCOPe data and create training datasets')
    parser.add_argument('json_file', type=Path, help='json file')
    parser.add_argument('output_folder', type=Path, help='Output folder for processed data')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--tm_score_threshold', type=float, default=0.6, help='TM-score threshold')
    parser.add_argument('--rmsd_threshold', type=float, default=5, help='RMSD threshold')
    parser.add_argument('--negative_start', type=int, default=1, help='Negative start')
    parser.add_argument('--negative_end', type=int, default=5, help='Negative end')
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)
    output_file = args.output_folder / (args.json_file.stem + ".tsv")
    if output_file.exists() and not args.overwrite:
        print(f"Skipping {output_file.stem} because it already exists")
    else:
        make_training_data_triplet_scop(
            json_file=args.json_file,
            output_file=output_file,
            negative_start=args.negative_start,
            negative_end=args.negative_end,
            overwrite=args.overwrite,
            tm_score_threshold=args.tm_score_threshold,
            rmsd_threshold=args.rmsd_threshold,
        )

if __name__ == "__main__":
    main()