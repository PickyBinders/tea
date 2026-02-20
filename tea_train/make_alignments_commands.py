import pandas as pd
from pathlib import Path
from collections import defaultdict

lookup_file = pd.read_csv("data/scope40_2_08/lookup_file.tsv", sep="\t", header=None, names=["id", "scop_id"])
lookup_file["superfamily"] = lookup_file["scop_id"].map(lambda x: ".".join(x.split(".")[:3]))
superfamily_to_ids = defaultdict(list)
for sid, level_id in zip(lookup_file["id"], lookup_file["superfamily"]):
    superfamily_to_ids[level_id].append(sid)
output_folder = Path("data/scope40_2_08/output")
matrices_folder = output_folder / "matrices"
matrices_folder.mkdir(exist_ok=True, parents=True)
pdb_folder = Path("data/scope40_2_08/pdbstyle-sel-gs-bib-40-2.08/pdbstyle-2.08")
with open("data/scope40_2_08/run_alignments.cmd", "w") as f:
    for superfamily in superfamily_to_ids:
        ids = superfamily_to_ids[superfamily]
        for i, id_1 in enumerate(ids):
            for id_2 in ids[i+1:]:
                f.write(f"python align_and_extract.py {id_1} {id_2} {pdb_folder} {matrices_folder}\n")
print("Generated command file: data/scope40_2_08/run_alignments.cmd")