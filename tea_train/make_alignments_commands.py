"""Write the original same-superfamily structural-alignment command list."""

from collections import defaultdict
from pathlib import Path
from shlex import join

from lightning.pytorch.cli import LightningArgumentParser


def write_commands(lookup_file, pdb_folder, alignment_folder, output):
    superfamilies = defaultdict(list)
    with Path(lookup_file).open() as handle:
        for line in handle:
            domain, scop = line.split()
            superfamilies[".".join(scop.split(".")[:3])].append(domain)

    Path(alignment_folder).mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as handle:
        for domains in superfamilies.values():
            domains = sorted(domains)
            for index, left in enumerate(domains):
                for right in domains[index + 1:]:
                    command = ["python", "-m", "tea_train.align_and_extract",
                               left, right, str(pdb_folder), str(alignment_folder)]
                    handle.write(join(command) + "\n")


def main():
    parser = LightningArgumentParser(description=__doc__)
    parser.add_argument("--lookup-file", type=Path, required=True)
    parser.add_argument("--pdb-folder", type=Path, required=True)
    parser.add_argument("--alignment-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_commands(args.lookup_file, args.pdb_folder, args.alignment_folder, args.output)
    print(f"Wrote structural-alignment commands to {args.output}")


if __name__ == "__main__":
    main()
