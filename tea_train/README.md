# TEA training

The current TEA model trains on full-chain sequence embeddings. Local EBA
alignments between chains in the same SCOP family provide positive examples.

Run from the TEA repository root in a Python environment with CUDA-enabled
PyTorch. Install TEA and the training dependencies.

```bash
python -m pip install -e .
python -m pip install -r tea_train/requirements.txt
```

## Prepare SCOPe40 training data

Provide SCOPe 2.08 parent-chain sequences in `chains.fasta`, and a whitespace-
delimited `lookup.tsv` with a domain ID and SCOP family per line. The source
`mapping.json` contains `context_fasta_sha256` and a `domains` mapping keyed by
domain ID. Each domain entry records its `domain_sequence`,
`domain_sequence_sha256`, mapping `identity` and `method`, and ordered
`slices` with `context_id`, `start`, and `stop`. Use zero-based, half-open
parent-chain coordinates. `folds.json` maps SCOP fold IDs to the four outer
folds used in the paper.

```bash
python -m tea_train.dataset import \
  --fasta chains.fasta --mapping mapping.json --lookup lookup.tsv \
  --root work --provenance work/import
python -m tea_train.dataset split --root work --seed 42 --historical-folds folds.json

raw=work/data/raw/scope40_2_08
splits=work/data/splits/scope40_2_08.json
spec=tea_train/configs/encoders/esm2-650m-4bit.json

python -m tea_train.embed --fasta "$raw/chains.fasta" --spec "$spec" \
  --output work/embeddings --allow-download --require-full-context
python -m tea_train.local_alignments --mapping "$raw/mapping.json" \
  --splits "$splits" --embeddings work/embeddings --output work/eba \
  --subset all --gap-open 8 --gap-extend 0.125
python -m tea_train.triplets --mapping "$raw/mapping.json" --splits "$splits" \
  --alignments work/eba --output work/triplets.parquet --subset all \
  --directions single --seed 42 --window 1 5 \
  --min-alignment-mean 2 --min-residue-score 2
```

On a cluster, run the encoder and trainer in a GPU allocation. For offline
embedding, stage the encoder in the Hugging Face cache before the job and
omit `--allow-download`.

## Train TEA

Partition the shared triplets for the four held-out folds. Generate the
paper configurations and run each with Lightning. The reference uses four
epochs and family-balanced sampling.

```bash
partition_folds() {
  for fold in 0 1 2 3; do
    for subset in fit validation; do
      python -m tea_train.partition_triplets --source "$1" \
        --output "$2/fold$fold/$subset.parquet" --mapping "$raw/mapping.json" \
        --splits "$splits" --outer-fold "$fold" --subset "$subset" || return
    done
  done
}
partition_folds work/triplets.parquet work
python -m tea_train.experiments --output work/ablations
python -m tea_train.fit fit --config work/ablations/selected_recipe/fold0/train.yaml
```

The no-uniform-loss and alphabet-size ablations use the same triplets. For
the far-window ablation, rerun `tea_train.triplets` with `--window 6 10` and
`--output work/far-window/triplets.parquet`, then run
`partition_folds work/far-window/triplets.parquet work/far-window`.
For each backbone comparison (`esm2-3b`, `prott5`, `prostt5`), use its
encoder config and rerun embedding, EBA and triplet generation under
`work/backbones/<backbone>/` with the same mapping and split. Use
`--require-full-context` for ESM2 and the encoder's window settings for T5.
Run `partition_folds` for each backbone's triplets. `work/ablations/runs.tsv`
lists the configuration and checkpoint paths for every fold and condition.

Train the released recipe on all chains with the full-data partition.

```bash
full_splits=work/data/splits/full.json
python -m tea_train.dataset full-training --root work --output "$full_splits"
python -m tea_train.partition_triplets --source work/triplets.parquet \
  --output work/full-data/fit.parquet --mapping "$raw/mapping.json" \
  --splits "$full_splits" --subset fit
python -m tea_train.fit fit --config tea_train/configs/train_full_chain.yaml \
  --data.splits "$full_splits" --data.outer_fold=null --data.data_folder work/full-data \
  --trainer.default_root_dir work/full-data/training
```

The [benchmark repository](https://github.com/PickyBinders/tea_benchmarks)
contains the figure notebooks and instructions for obtaining their inputs.

## Reproduce TEA-v0

TEA-v0 uses US-align supervision, TM-score and RMSD filtering, and
multi-negative sampling. Use `tea_train.make_alignments_commands` to generate
the domain-pair commands, `tea_train.make_triplets` to sample triplets from
the results, and `tea_train.make_parquet` to assemble the fold data.
The trainer accepts FP16 domain caches and `PREFIX_embeddings_*.pt` files.

```bash
python -m tea_train.train_tea fit --config tea_train/configs/train_legacy_structural.yaml
```

[`config.yaml`](config.yaml) records the released TEA-v0 recipe. The bundled
`train_legacy_structural.yaml` recipe uses PyTorch AdamW.
