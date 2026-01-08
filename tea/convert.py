from pathlib import Path
from biotite.sequence.io import fasta
import re
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from collections import defaultdict
from .model import Tea
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LENGTH_BINS = [
    50,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    2000,
    3000,
    4000,
    5000,
]
BATCH_SIZES = {
    50: 1024,
    100: 1024,
    200: 512,
    300: 512,
    400: 256,
    500: 256,
    600: 256,
    700: 128,
    800: 128,
    900: 128,
    1000: 128,
    2000: 32,
    3000: 16,
    4000: 8,
    5000: 4,
}

GPU_TIME_ESTIMATES = {
    50: 0.0048788634128868,
    100: 0.0096358241979032,
    200: 0.0195782056078314,
    300: 0.0300744888372719,
    400: 0.0410776803269982,
    500: 0.051343567110598,
    600: 0.0632535872980952,
    700: 0.0756181448698043,
    800: 0.0886568557471036,
    900: 0.1022435456514358,
    1000: 0.1136764369904995,
    2000: 0.2711546048521995,
    3000: 0.4787218451499939,
    4000: 0.7247123420238495,
    5000: 1.0142204642295838,
}


def run_batch(
    sequences,
    tokenizer,
    esm2,
    tea,
    save_avg_entropy=False,
    save_logits=False,
    save_residue_entropy=False,
):
    device = next(tea.parameters()).device
    try:
        spaced_seqs = [
            " ".join(list(re.sub(r"[UZOBJ]", "X", seq.replace("-", ""))) for _, seq in sequences
        ]
        batch = tokenizer.batch_encode_plus(
            spaced_seqs, add_special_tokens=True, padding="longest"
        )
        if len(batch) == 0:
            return None
        batch_tokens = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        with torch.no_grad():
            embeddings = esm2(
                input_ids=batch_tokens, attention_mask=attention_mask
            ).last_hidden_state.to(device)
            results = tea.to_sequences(
                embeddings=embeddings,
                input_ids=batch_tokens,
                return_avg_entropy=save_avg_entropy,
                return_logits=save_logits,
                return_residue_entropy=save_residue_entropy,
            )
        if not save_avg_entropy and not save_logits and not save_residue_entropy:
            yield [{"sequence": r} for r in results]
        else:
            keys = ["sequences"]
            new_keys = ["sequence"]
            if save_avg_entropy:
                keys.append("avg_entropy")
                new_keys.append("avg_entropy")
            if save_logits:
                keys.append("logits")
                new_keys.append("logits")
            if save_residue_entropy:
                keys.append("residue_entropy")
                new_keys.append("residue_entropy")
            results_list = []
            for i in range(len(sequences)):
                result_dict = {}
                for key, new_key in zip(keys, new_keys):
                    result_dict[new_key] = results[key][i]
                results_list.append(result_dict)
            yield results_list
    except (torch.cuda.OutOfMemoryError, MemoryError) as e:
        if "cuda" in device.type:
            torch.cuda.empty_cache()
        error_type = "CUDA" if isinstance(e, torch.cuda.OutOfMemoryError) else "CPU"
        if len(sequences) == 1:
            logger.error(
                f"{error_type} out of memory error for single sequence. "
                f"Sequence is too large to process even with batch size 1. "
                f"Skipping sequence: {sequences[0][0]}"
            )
            return
        new_batch_size = len(sequences) // 2
        logger.info(
            f"{error_type} out of memory error for batch size {len(sequences)}. Running with batch size divided by 2 ({new_batch_size})"
        )
        for i in range(0, len(sequences), new_batch_size):
            yield from run_batch(  # Use yield from for recursive calls
                sequences[i : i + new_batch_size],
                tokenizer,
                esm2,
                tea,
                save_avg_entropy,
                save_logits,
                save_residue_entropy,
            )


def convert_sequences(
    fasta_file,
    output_file,
    tokenizer,
    esm2,
    tea,
    save_logits=False,
    save_avg_entropy=True,
    save_residue_entropy=False,
    lowercase_entropy=True,
    entropy_lowercase_threshold=0.3,
):
    length_groups = defaultdict(list)
    num_sequences = 0
    for header, seq in fasta.FastaFile.read(fasta_file).items():
        found = False
        seq_len = len(seq)
        for bin_len in LENGTH_BINS:
            if seq_len <= bin_len:
                length_groups[bin_len].append((header, seq))
                found = True
                break
        if not found:
            length_groups[10000].append((header, seq))
        num_sequences += 1
    buffer_time = 60  # 1 minute buffer
    total_time_estimate = (
        sum(
            GPU_TIME_ESTIMATES.get(seq_len, 1) * len(group)
            for seq_len, group in length_groups.items()
        )
        + buffer_time
    )
    device = next(tea.parameters()).device
    if "cuda" in device.type:
        if total_time_estimate < 3600:  # Less than an hour
            time_str = f"{total_time_estimate / 60:.2f} minutes"
        else:  # An hour or more
            time_str = f"{total_time_estimate / 3600:.2f} hours"
        logger.info(f"Estimated time to complete conversion: {time_str}")
    logits_dict = dict()
    residue_entropy_dict = dict()
    logger.info(
        f"Processing {num_sequences} sequences in total, across {len(length_groups)} length groups"
    )
    disable_tqdm = logger.getEffectiveLevel() > logging.INFO

    with open(output_file, "w") as f:
        for s, (seq_len, group) in enumerate(length_groups.items()):
            batch_size = BATCH_SIZES.get(seq_len, 1)
            total_batches = (len(group) + batch_size - 1) // batch_size
            for i in tqdm(
                range(0, len(group), batch_size),
                desc=f"Max. length {seq_len}",
                unit="batch",
                total=total_batches,
                leave=True,
                disable=disable_tqdm,
            ):
                headers = [h for h, _ in group[i : i + batch_size]]
                for results_batch in run_batch(
                    group[i : i + batch_size],
                    tokenizer,
                    esm2,
                    tea,
                    save_avg_entropy,
                    save_logits,
                    save_residue_entropy | lowercase_entropy,
                ):
                    for header, result in zip(headers, results_batch):
                        if lowercase_entropy:
                            result["sequence"] = "".join(
                                [
                                    s if e < entropy_lowercase_threshold else s.lower()
                                    for s, e in zip(
                                        result["sequence"], result["residue_entropy"]
                                    )
                                ]
                            )
                        if save_avg_entropy:
                            f.write(
                                f">{header}|H={result['avg_entropy']:.3f}\n{result['sequence']}\n"
                            )
                        else:
                            f.write(f">{header}\n{result['sequence']}\n")
                        if save_logits:
                            logits_dict[header] = result["logits"]
                        if save_residue_entropy:
                            residue_entropy_dict[header] = result["residue_entropy"]
    if save_logits:
        torch.save(logits_dict, output_file.parent / f"{output_file.stem}_logits.pt")
    if save_residue_entropy:
        torch.save(
            residue_entropy_dict,
            output_file.parent / f"{output_file.stem}_residue_entropy.pt",
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fasta_file",
        type=Path,
        required=True,
        help="Input FASTA file containing protein amino acid sequences",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        required=True,
        help="Output FASTA file for generated tea sequences",
    )
    parser.add_argument(
        "-l",
        "--save_logits",
        action="store_true",
        help="Save per-residue logits to .pt file",
    )
    parser.add_argument(
        "-H",
        "--save_avg_entropy",
        action="store_true",
        help="Save average entropy values in FASTA identifiers",
    )
    parser.add_argument(
        "-r",
        "--save_residue_entropy",
        action="store_true",
        help="Save per-residue entropy values to .pt file",
    )
    parser.add_argument(
        "-c",
        "--lowercase_entropy",
        action="store_true",
        help="Save residues with entropy > threshold in lowercase",
    )
    parser.add_argument(
        "-t",
        "--entropy_threshold",
        type=float,
        default=0.25,
        help="Entropy threshold for lowercase conversion",
    )

    args = parser.parse_args()
    assert not args.output_file.exists(), (
        f"Output file {args.output_file} already exists, refusing to overwrite"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from PickyBinders/tea")
    tea = Tea.from_pretrained("PickyBinders/tea").to(device)
    tea.eval()
    logger.info("Loading model from facebook/esm2_t33_650M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    esm2 = AutoModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D",
        dtype="auto",
        quantization_config=bnb_config,
        add_pooling_layer=False,
    ).to(device)
    esm2.eval()
    logger.info(f"Converting sequences from {args.fasta_file} to {args.output_file}")
    convert_sequences(
        args.fasta_file,
        args.output_file,
        tokenizer,
        esm2,
        tea,
        args.save_logits,
        args.save_avg_entropy,
        args.save_residue_entropy,
        args.lowercase_entropy,
        args.entropy_threshold,
    )
    logger.info("Conversion complete")
    message = f"Saved sequences to {args.output_file}"
    if args.lowercase_entropy:
        message += f"\nLowercased letters have entropy > {args.entropy_threshold}"
    if args.save_logits:
        message += f"\nSaved logits to {args.output_file.parent / f'{args.output_file.stem}_logits.pt'}"
    if args.save_residue_entropy:
        message += f"\nSaved residue entropy to {args.output_file.parent / f'{args.output_file.stem}_residue_entropy.pt'}"
    logger.info(message)


if __name__ == "__main__":
    main()
