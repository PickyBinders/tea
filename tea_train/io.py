"""FASTA, JSON and split files used by TEA training."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from biotite.sequence.io.fasta import FastaFile


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_fasta(path):
    """Read complete sequences with unique FASTA identifiers."""
    records = {}
    # Preflight duplicates before the mapping-like library reader can hide one.
    identifiers = set()
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                fields = line[1:].split()
                if not fields or fields[0] in identifiers:
                    raise ValueError("Duplicate or empty FASTA identifier")
                identifiers.add(fields[0])
    for header, sequence in FastaFile.read(path).items():
        identifier = header.split()[0]
        if not identifier or identifier in records:
            raise ValueError(f"Duplicate or empty FASTA identifier: {identifier}")
        if not sequence or not sequence.isascii() or not sequence.isalpha():
            raise ValueError(f"Invalid or empty sequence: {identifier}")
        records[identifier] = sequence
    if not records:
        raise ValueError("Empty FASTA")
    return records


def write_json(path, value):
    """Atomic writes prevent interrupted runs from looking complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = handle.name
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def partition_ids(partitions, subset, outer_fold=None):
    """Read one named ID set from a CV split or standalone cohort."""
    if "identifiers" in partitions:
        declared_fold = partitions.get("outer_fold")
        if outer_fold != declared_fold:
            raise ValueError("The requested fold differs from the cohort")
        expected = "tuning" if "tuning" in partitions.get("purpose", "") else "all"
        if subset != expected:
            raise ValueError(f"This standalone cohort uses {expected}")
        return sorted(partitions["identifiers"])
    if outer_fold is None:
        try:
            return sorted(partitions["partitions"][subset])
        except KeyError as error:
            raise ValueError("This partition requires an outer fold") from error
    try:
        return sorted(partitions["runs"][str(outer_fold)]["partitions"][subset])
    except KeyError as error:
        raise ValueError("Unknown fold or partition") from error


