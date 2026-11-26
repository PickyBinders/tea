"""Create the structural triplets used by the original TEA training recipe."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from biotite.sequence.align import Alignment
from lightning.pytorch.cli import LightningArgumentParser


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


def sample_pair(data, negative_start=1, negative_end=5,
                tm_score_threshold=0.6, rmsd_threshold=5.0):
    """Yield legacy multi-negative triplets from one US-align record."""
    if data["tm_score_min"] < tm_score_threshold:
        return
    id_1, id_2 = data["id_1"], data["id_2"]
    aligned = [data["aln"][id_1], data["aln"][id_2]]
    trace = Alignment.trace_from_strings(aligned)
    lengths = [len(sequence.replace("-", "")) for sequence in aligned]

    for column, (index_1, positive) in enumerate(trace):
        rmsd = data["rmsds"][column]
        if index_1 == -1 or positive == -1 or rmsd is None or rmsd > rmsd_threshold:
            continue
        for shift in range(-negative_end, negative_end + 1):
            if abs(shift) < negative_start:
                continue
            other_column = column + shift
            if not 0 <= other_column < len(trace):
                continue
            negative = trace[other_column, 1]
            if negative == -1:
                continue
            yield ResidueTriplet(
                protein_1=f"{id_1}/1-{lengths[0]}",
                protein_2=f"{id_2}/1-{lengths[1]}",
                index_1=int(index_1),
                index_2_positive=int(positive),
                index_2_negative=int(negative),
                shift=shift,
                aa1=aligned[0][column],
                aa2_positive=aligned[1][column],
                aa2_negative=aligned[1][other_column],
                rmsd=float(rmsd),
            )


sample_from_protein_pair_scop = sample_pair


def make_triplets(json_file, output_file, negative_start=1, negative_end=5,
                  tm_score_threshold=0.6, rmsd_threshold=5.0):
    with Path(json_file).open() as handle:
        data = json.load(handle)
    rows = []
    for triplet in sample_pair(
        data, negative_start, negative_end, tm_score_threshold, rmsd_threshold
    ):
        row = asdict(triplet)
        row["protein_1"], row["range_1"] = row["protein_1"].split("/", 1)
        row["protein_2"], row["range_2"] = row["protein_2"].split("/", 1)
        rows.append(row)
    columns = [
        "protein_1", "protein_2", "range_1", "range_2", "index_1",
        "index_2_positive", "index_2_negative", "shift", "aa1",
        "aa2_positive", "aa2_negative", "rmsd",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_file, sep="\t", index=False)
    return len(rows)


def make_training_data_triplet_scop(
    json_file,
    output_file,
    negative_start=1,
    negative_end=10,
    overwrite=False,
    tm_score_threshold=0.6,
    rmsd_threshold=5.0,
):
    """Retain the original structural-triplet library entry point."""
    output_file = Path(output_file)
    if output_file.exists() and not overwrite:
        return 0
    return make_triplets(
        json_file,
        output_file,
        negative_start,
        negative_end,
        tm_score_threshold,
        rmsd_threshold,
    )


def main():
    parser = LightningArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tm-score-threshold", type=float, default=0.6)
    parser.add_argument("--rmsd-threshold", type=float, default=5.0)
    parser.add_argument("--negative-start", type=int, default=1)
    parser.add_argument("--negative-end", type=int, default=5)
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)
    output = args.output_folder / f"{args.json_file.stem}.tsv"
    if output.exists() and not args.overwrite:
        print(f"Skipping existing {output}")
        return
    count = make_triplets(
        args.json_file,
        output,
        args.negative_start,
        args.negative_end,
        args.tm_score_threshold,
        args.rmsd_threshold,
    )
    print(f"Wrote {count} legacy structural triplets to {output}")


if __name__ == "__main__":
    main()
