from multiprocessing import Pool
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path

data = pd.concat([pd.read_csv(f, sep="\t") for f in tqdm(Path("data/scope40_2_08/triplets").glob("*.tsv"))])
lookup_file = pd.read_csv("data/scope40_2_08/lookup_file.tsv", sep="\t", header=None, names=["id", "scop_id"])
lookup_file["fold"] = lookup_file["scop_id"].map(lambda x: ".".join(x.split(".")[:2]))
lookup_file["superfamily"] = lookup_file["scop_id"].map(lambda x: ".".join(x.split(".")[:3]))
lookup_file["family"] = lookup_file["scop_id"].map(lambda x: ".".join(x.split(".")[:4]))
lookup_file["architecture"] = lookup_file["scop_id"].map(lambda x: x.split(".")[0])
id_to_scop = dict(zip(lookup_file["id"], lookup_file["scop_id"]))
id_to_level = {}
for level in ["fold", "superfamily", "family", "architecture"]:
    id_to_level[level] = dict(zip(lookup_file["id"], lookup_file[level]))

data["scop_id_1"] = data["protein_1"].map(lambda x: id_to_scop[x])
data["scop_id_2"] = data["protein_2"].map(lambda x: id_to_scop[x])
data["abs_shift"] = data["shift"].abs()


with open("data/scope40_2_08/fold_split.json", "r") as f:
    fold_split = json.load(f)

data["fold_split"] = data["protein_1"].map(lambda x: fold_split[id_to_level["fold"][x]])
data = data[data["abs_shift"] <= 5].sample(frac=1.0, random_state=42).groupby(["protein_1", "protein_2", "index_1", "index_2_positive"]).head(1).reset_index(drop=True)
data = data[["protein_1", "protein_2", "index_1", "index_2_positive", "index_2_negative", "fold_split"]].reset_index(drop=True)
data = data.groupby(['protein_1', 'protein_2', 'fold_split']).agg({
    'index_1': list,
    'index_2_positive': list, 
    'index_2_negative': list
}).apply(lambda x: {
    'index_1': x['index_1'],
    'index_2_positive': x['index_2_positive'],
    'index_2_negative': x['index_2_negative']
}, axis=1).rename('triplet_data').reset_index()

folder = Path("data")
data.to_parquet(folder / "train.parquet", index=False)
data[(data["fold_split"] == 0)].to_parquet(folder / "val.parquet", index=False)
