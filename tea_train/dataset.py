"""Import full-chain sequence data and make component-disjoint SCOP splits."""

from lightning.pytorch.cli import LightningArgumentParser
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import shutil

from scipy.cluster.hierarchy import DisjointSet

from tea_train.io import read_fasta, read_json, sha256_file, stable_hash, write_json


def normalized_sequence(sequence):
    # Match encoder unknown-residue policy when detecting identical inputs.
    return sequence.upper().translate(str.maketrans({c: "X" for c in "UZOBJ"}))


def read_lookup(path):
    result = {}
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2 or len(fields[1].split(".")) != 4:
                raise ValueError(f"Invalid SCOP lookup row {number}")
            domain, family = fields
            if domain in result:
                raise ValueError(f"Duplicate domain label: {domain}")
            result[domain] = family
    return result


def clean_mapping(sequences, source, labels):
    """Export the portable sequence and annotation fields."""
    domains = {}
    chains = {key: {"length": len(seq), "sequence_sha256": hashlib.sha256(
        normalized_sequence(seq).encode()).hexdigest(), "domains": [], "families": []}
        for key, seq in sequences.items()}
    if source.get("context_fasta_sha256") is None:
        raise ValueError("Source mapping lacks sequence checksum")
    for identifier, entry in sorted(source["domains"].items()):
        if identifier not in labels:
            raise ValueError(f"Missing SCOP label: {identifier}")
        family = labels[identifier]
        original = entry["domain_sequence"].upper()
        if hashlib.sha256(entry["domain_sequence"].encode()).hexdigest() != entry["domain_sequence_sha256"]:
            raise ValueError(f"Domain sequence checksum mismatch: {identifier}")
        pieces, slices, occupied = [], [], set()
        for region in entry["slices"]:
            chain, start, stop = region["context_id"], region["start"], region["stop"]
            if chain not in sequences or not isinstance(start, int) or not isinstance(stop, int):
                raise ValueError(f"Invalid chain/slice: {identifier}")
            if not 0 <= start < stop <= len(sequences[chain]):
                raise ValueError(f"Out-of-bounds slice: {identifier}")
            coordinates = {(chain, index) for index in range(start, stop)}
            if coordinates & occupied:
                raise ValueError(f"Overlapping domain slices: {identifier}")
            occupied.update(coordinates)
            pieces.append(sequences[chain][start:stop].upper())
            slices.append({"chain_id": chain, "start": start, "stop": stop})
            chains[chain]["domains"].append(identifier)
            chains[chain]["families"].append(family)
        mapped = "".join(pieces)
        if len(mapped) != len(original):
            raise ValueError(f"Domain mapping length mismatch: {identifier}")
        # Source identity treats unknown residues as compatible.
        identity = sum(a == b or "X" in (a, b) for a, b in zip(mapped, original)) / len(original)
        if identity + 1e-8 < float(entry["identity"]):
            raise ValueError(f"Domain mapping identity below source claim: {identifier}")
        domains[identifier] = {"family": family, "superfamily": ".".join(family.split(".")[:3]),
            "fold": ".".join(family.split(".")[:2]), "slices": slices,
            "domain_sequence": original, "mapping_identity": identity,
            "mapping_method": entry["method"]}
    for identifier, chain in chains.items():
        if not chain["domains"]:
            raise ValueError(f"Chain has no mapped annotations: {identifier}")
        chain["domains"] = sorted(set(chain["domains"]))
        chain["families"] = sorted(set(chain["families"]))
        chain["superfamilies"] = sorted({".".join(f.split(".")[:3]) for f in chain["families"]})
        chain["folds"] = sorted({".".join(f.split(".")[:2]) for f in chain["families"]})
    return {"schema_version": 1, "coordinates": "zero-based-half-open",
            "chains": chains, "domains": domains}


def import_dataset(fasta, mapping, lookup, root, provenance):
    root, provenance = Path(root).resolve(), Path(provenance).resolve()
    if provenance == root / "data" or root / "data" in provenance.parents:
        raise ValueError("Provenance must be outside data")
    raw = root / "data/raw/scope40_2_08"
    manifest = root / "manifests/scope40_2_08.json"
    if manifest.exists() or raw.exists():
        raise FileExistsError("Dataset import already exists; choose a new root/version")
    source = read_json(mapping)
    if sha256_file(fasta) != source["context_fasta_sha256"]:
        raise ValueError("FASTA does not match source mapping checksum")
    sequences = read_fasta(fasta)
    clean = clean_mapping(sequences, source, read_lookup(lookup))
    raw.mkdir(parents=True)
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fasta, raw / "chains.fasta")
    write_json(raw / "mapping.json", clean)
    imports = {}
    for name, path in (("context.fasta", fasta), ("context.original.json", mapping), ("lookup.tsv", lookup)):
        destination = provenance / name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(path, destination)
        imports[name] = {"source": str(Path(path).resolve()), "sha256": sha256_file(path)}
    write_json(provenance / "import.json", {"schema_version": 1, "sources": imports})
    info = {"schema_version": 1, "dataset": "scope40_2_08_full_chains",
        "sequence_definition": "full deposited PDB chain context",
        "coordinates": clean["coordinates"], "chains": len(sequences),
        "domains": len(clean["domains"]), "residues": sum(map(len, sequences.values())),
        "files": {str(path.relative_to(root)): sha256_file(path)
                  for path in (raw / "chains.fasta", raw / "mapping.json")}}
    info["dataset_id"] = stable_hash(info)
    write_json(manifest, info)
    return info


def components(mapping):
    chains = mapping["chains"]
    union = DisjointSet(sorted(chains))
    seen = {}
    for identifier, entry in sorted(chains.items()):
        keys = [("fold", fold) for fold in entry["folds"]]
        keys.append(("sequence", entry["sequence_sha256"]))
        for key in keys:
            if key in seen:
                union.merge(identifier, seen[key])
            else:
                seen[key] = identifier
    # Discontinuous domains spanning chains must also stay together.
    for domain in mapping["domains"].values():
        ids = [region["chain_id"] for region in domain["slices"]]
        for other in ids[1:]:
            union.merge(ids[0], other)
    groups = defaultdict(list)
    for identifier in sorted(chains):
        groups[union[identifier]].append(identifier)
    result = []
    for ids in groups.values():
        folds = sorted({fold for key in ids for fold in chains[key]["folds"]})
        families = Counter(family for key in ids for family in chains[key]["families"])
        result.append({"id": stable_hash(ids)[:16], "chains": ids, "folds": folds,
            "weights": [len(ids), sum(chains[key]["length"] for key in ids),
                        sum(n * (n - 1) // 2 for n in families.values()), len(folds)]})
    return sorted(result, key=lambda item: item["id"])


def balanced_assign(groups, fractions, seed):
    """Deterministic greedy packing over chains/residues/family-pairs/folds."""
    if len(groups) < len(fractions) or any(f <= 0 for f in fractions):
        raise ValueError("Not enough components or invalid partition fractions")
    total = [max(1, sum(g["weights"][i] for g in groups)) for i in range(4)]
    ordered = sorted(groups, key=lambda g: (
        -max(w / t for w, t in zip(g["weights"], total)), stable_hash([seed, g["id"]])))
    loads = [[0.0] * 4 for _ in fractions]
    assignments = {}
    for group in ordered:
        weights = [w / t for w, t in zip(group["weights"], total)]
        index = min(range(len(fractions)), key=lambda j: (
            sum(((loads[j][i] + weights[i]) ** 2 - loads[j][i] ** 2) / fractions[j]
                for i in range(4)), j))
        assignments[group["id"]] = index
        loads[index] = [a + b for a, b in zip(loads[index], weights)]
    if set(assignments.values()) != set(range(len(fractions))):
        raise ValueError("Packing produced an empty partition")
    return assignments


def audit_partitions(mapping, partitions):
    seen = {"chain": {}, "fold": {}, "sequence": {}}
    counts = {}
    for name, identifiers in partitions.items():
        counts[name] = len(identifiers)
        for key in identifiers:
            entry = mapping["chains"][key]
            values = {"chain": [key], "fold": entry["folds"], "sequence": [entry["sequence_sha256"]]}
            for kind, keys in values.items():
                for value in keys:
                    if value in seen[kind] and seen[kind][value] != name:
                        raise ValueError(f"{kind} leakage between {seen[kind][value]} and {name}: {value}")
                    seen[kind][value] = name
    if set(seen["chain"]) != set(mapping["chains"]):
        raise ValueError("Partitions do not cover every chain")
    return {"chains": counts, "fold_overlap": 0, "sequence_overlap": 0, "chain_overlap": 0}


def make_splits(mapping, seed=42, validation_fraction=0.15, historical_folds=None):
    if not 0 < validation_fraction < 0.5:
        raise ValueError("Inner validation fraction must be between zero and 0.5")
    original_count = len(mapping["chains"])
    excluded = {}
    if historical_folds is not None:
        # Keep whole chains in one outer fold.
        for key, chain in mapping["chains"].items():
            values = {historical_folds[fold] for fold in chain["folds"]}
            if len(values) != 1:
                excluded[key] = "chain-spans-outer-folds"
        hashes = defaultdict(list)
        for key, chain in mapping["chains"].items():
            hashes[chain["sequence_sha256"]].append(key)
        for ids in hashes.values():
            values = {historical_folds[fold] for key in ids for fold in mapping["chains"][key]["folds"]}
            if len(values) != 1:
                for key in ids:
                    excluded.setdefault(key, "identical-sequence-spans-outer-folds")
        mapping = {"chains": {key: value for key, value in mapping["chains"].items() if key not in excluded},
                   "domains": {key: value for key, value in mapping["domains"].items()
                               if all(region["chain_id"] not in excluded for region in value["slices"])}}
    groups = components(mapping)
    largest = max(len(group["chains"]) for group in groups)
    if largest > 0.45 * len(mapping["chains"]):
        raise ValueError(f"Largest indivisible component contains {largest}/{len(mapping['chains'])} chains; "
                         "balanced four-fold CV is infeasible. Use an explicit purged historical split.")
    assignment = (balanced_assign(groups, [0.25] * 4, seed) if historical_folds is None else
                  {group["id"]: historical_folds[group["folds"][0]] for group in groups})
    if historical_folds is not None and any(len({historical_folds[f] for f in group["folds"]}) != 1 for group in groups):
        raise ValueError("Purged component still bridges historical folds")
    output = {"schema_version": 1, "seed": seed, "validation_fraction": validation_fraction,
              "strategy": "connected-components" if historical_folds is None else "historical-folds-purged-chains",
              "original_chains": original_count, "retained_chains": len(mapping["chains"]),
              "excluded_chains": excluded, "components": groups, "outer_assignments": assignment, "runs": {}}
    for outer in range(4):
        train_groups = [g for g in groups if assignment[g["id"]] != outer]
        inner = balanced_assign(train_groups, [1 - validation_fraction, validation_fraction], seed + outer)
        partition = {"fit": [], "validation": [], "test": []}
        for group in groups:
            subset = "test" if assignment[group["id"]] == outer else ("fit" if inner[group["id"]] == 0 else "validation")
            partition[subset].extend(group["chains"])
        partition = {key: sorted(value) for key, value in partition.items()}
        audit = audit_partitions(mapping, partition)
        if any(not identifiers for identifiers in partition.values()):
            raise ValueError("Empty partition")
        membership = {chain: subset for subset, ids in partition.items() for chain in ids}
        domain_partitions = {subset: [] for subset in partition}
        for domain, entry in sorted(mapping["domains"].items()):
            subsets = {membership[region["chain_id"]] for region in entry["slices"]}
            if len(subsets) != 1:
                raise ValueError(f"Domain spans partitions: {domain}")
            domain_partitions[next(iter(subsets))].append(domain)
        output["runs"][str(outer)] = {"partitions": partition, "domain_partitions": domain_partitions, "audit": audit}
    output["split_id"] = stable_hash(output)
    return output


def full_training_partition(mapping, dataset_id):
    result = {"schema_version": 1, "purpose": "full-dataset-training",
              "dataset_id": dataset_id, "mapping_id": stable_hash(mapping),
              "partitions": {"fit": sorted(mapping["chains"])}}
    if not result["partitions"]["fit"]:
        raise ValueError("Full-data training requires imported chains")
    result["split_id"] = stable_hash(result)
    return result


def main():
    parser = LightningArgumentParser(description=__doc__)
    commands = parser.add_subcommands(required=True)
    importer = LightningArgumentParser(description="Import sequence data with separate provenance")
    for name in ("fasta", "mapping", "lookup", "root", "provenance"):
        importer.add_argument(f"--{name}", type=Path, required=True)
    commands.add_subcommand("import", importer)
    splitter = LightningArgumentParser(description="Build leakage-safe full-chain splits")
    splitter.add_argument("--root", type=Path, required=True)
    splitter.add_argument("--seed", type=int, default=42)
    splitter.add_argument("--validation-fraction", type=float, default=0.15)
    splitter.add_argument("--historical-folds", type=Path,
                          help="Explicitly purge cross-fold chains while preserving this fold assignment")
    commands.add_subcommand("split", splitter)
    full = LightningArgumentParser(description="Declare all imported chains for main-model fitting after CV recipe selection")
    full.add_argument("--root", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    commands.add_subcommand("full-training", full)
    parsed = parser.parse_args()
    command = parsed.subcommand
    args = getattr(parsed, command)
    if command == "import":
        info = import_dataset(args.fasta, args.mapping, args.lookup, args.root, args.provenance)
        print(f"Imported {info['chains']} chains and {info['domains']} domains; dataset {info['dataset_id']}")
    else:
        manifest = read_json(args.root / "manifests/scope40_2_08.json")
        for relative, checksum in manifest["files"].items():
            if sha256_file(args.root / relative) != checksum:
                raise ValueError(f"Dataset checksum mismatch: {relative}")
        mapping = read_json(args.root / "data/raw/scope40_2_08/mapping.json")
        if command == "full-training":
            if args.output.exists():
                raise FileExistsError("Never overwrite a full-data training partition")
            result = full_training_partition(mapping, manifest["dataset_id"])
            write_json(args.output, result)
            print(f"Declared {len(result['partitions']['fit'])} full-data fitting chains")
            return
        result = make_splits(mapping, args.seed, args.validation_fraction,
                             read_json(args.historical_folds) if args.historical_folds else None)
        result["dataset_id"] = manifest["dataset_id"]
        write_json(args.root / "data/splits/scope40_2_08.json", result)
        print(f"Split {len(result['components'])} connected components; split {result['split_id']}")
        for key, run in result["runs"].items():
            print(f"Outer fold {key}: {run['audit']}")


if __name__ == "__main__":
    main()
