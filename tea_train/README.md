# How to train your TEA

Framework for training TEA from sequences and structures using the SCOPe40 dataset.

## Installation

```bash
pip install -e ".[training]"
```

## Generating training data

### 1. Download SCOPe40 structures

Download and extract SCOPe40 structures:

```bash
wget https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz -O data/scope40_2_08/pdbstyle-sel-gs-bib-40-2.08.tgz
tar -xzf data/scope40_2_08/pdbstyle-sel-gs-bib-40-2.08.tgz -C data/scope40_2_08/
```

### 2. Generate embeddings

Compute per-residue ESM-2 embeddings (4-bit quantized) for each sequence:

```bash
python generate_embeddings.py \
    --sequence_file data/scope40_2_08/scope40_2_08_from_alignments.fasta \
    --output_prefix data/embeddings/esm2_4bit \
    --quantize
```

### 3. Make alignments

Generate pairwise structural alignments within each superfamily:

```bash
python make_alignments_commands.py
```

This produces a command file (`run_alignments.cmd`) that can be run with whichever scheduler or parallelization strategy you prefer.

### 4. Make triplets

Use the alignments to generate training triplets of residues:

```bash
python make_triplets_commands.py
```

This produces a command file (`make_triplets.cmd`) that can be run with whichever scheduler or parallelization strategy you prefer. Each line calls `make_triplets.py` on one alignment JSON file, for example:

```bash
python make_triplets.py data/scope40_2_08/matrices/a.1.1.json data/scope40_2_08/triplets \
    --negative_start 1 --negative_end 5 --tm_score_threshold 0.6 --rmsd_threshold 5
```

### 5. Build training and validation splits

Consolidate the triplets into Parquet files split by SCOPe fold:

```bash
python make_parquet.py
```

This produces `train.parquet` and `val.parquet`.

## Training the model

Once you have the embeddings and the Parquet splits, run:

```bash
python train_tea.py fit -c config.yaml
```