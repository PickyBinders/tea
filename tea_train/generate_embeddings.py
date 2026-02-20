import torch
from pathlib import Path
import argparse
from typing import List, Tuple, Iterator
import re
from tqdm import tqdm
from biotite.sequence.io import fasta
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from transformers import BitsAndBytesConfig
from torch.nn import functional as F


def split_sequence(sequence: str, max_length: int, overlap: int = 0) -> List[str]:
    """Split a sequence into overlapping chunks.
    
    Args:
        sequence (str): The input sequence to split
        max_length (int): Maximum length for each chunk
        overlap (int): Number of residues to overlap between chunks
        
    Returns:
        List[str]: List of sequence chunks with overlap
    """
    if len(sequence) <= max_length:
        return [sequence]
    
    # Calculate number of chunks needed (ceiling division)
    seq_len = len(sequence)
    n_chunks = (seq_len + max_length - 1) // max_length
    
    chunks = []
    for i in range(n_chunks):
        if i == 0:  # First chunk
            # Include overlap at the end
            chunk = sequence[:max_length + overlap]
        elif i == n_chunks - 1:  # Last chunk
            # Include overlap at the start
            chunk = sequence[i * max_length - overlap:]
        else:  # Middle chunks
            # Include overlap on both sides
            start = i * max_length - overlap
            end = (i + 1) * max_length + overlap
            chunk = sequence[start:end]
        
        chunks.append(chunk)
    
    return chunks

def parse_fasta(fasta_file: Path, max_length: int = 1024, overlap: int = 128) -> Iterator[Tuple[str, str]]:
    valid_aas = set('ACDEFGHIKLMNPQRSTVWYX')
    split_count = 0
    for header, sequence in fasta.FastaFile.read(fasta_file).items():
        sequence = ''.join('X' if c not in valid_aas else c for c in sequence)
        if len(sequence) > max_length:
            split_count += 1
            chunks = split_sequence(sequence, max_length, overlap)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    yield (header, chunk)
                else:
                    yield (f"{header}_chunk{i+1}of{len(chunks)}", chunk)
        else:
            yield (header, sequence)

def tokenize_sequences(model, sequence_file, max_length: int = 1024, overlap: int = 128, trim_start: int = 1, trim_end: int = 1, mask_prob: float = 0):
    # Load ESM-2 model and alphabet
    if "prot_t5" in model:
        tokenizer = T5Tokenizer.from_pretrained(model, torch_dtype=torch.bfloat16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.bfloat16, local_files_only=True)
    tokenized_sequences = {}
    masked_indices = {}
    for protein, seq in tqdm(
        parse_fasta(sequence_file, max_length, overlap), desc="Tokenizing sequences"
    ):
        if len(seq) > max_length:
            continue
        spaced_seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        tokens = tokenizer.encode(
            spaced_seq,
            return_tensors="pt",
        )
        tokens = tokens.squeeze(0)
        if mask_prob > 0:
            mask = torch.rand(tokens.size(0)) < mask_prob
            tokens[mask] = tokenizer.mask_token_id
            masked_indices[protein] = torch.where(mask)[0]
        tokenized_sequences[protein] = tokens
        assert len(tokenized_sequences[protein]) == len(seq) + trim_start + trim_end, f"Sequence length mismatch for {protein}: {len(tokenized_sequences[protein])} != {len(seq)} + {trim_start} + {trim_end}"
    return tokenized_sequences, masked_indices
            

def embed_sequences(model, sequence_file, output_prefix, max_length: int = 1024, overlap: int = 128, batch_size=8, chunk_size=2500, padding_idx=1, trim_start=1, trim_end=1, mask_prob=0, quantize=True):
    """
    Precompute embeddings for all tokenized sequences and save to a file.
    Args:
        sequence_file: Path to FASTA file with sequences
        output_file: Path to save embeddings and tokens
        max_length: Maximum sequence length before splitting
        overlap: Overlap size for sequence chunks
        batch_size: Batch size for embedding computation
        chunk_size: Number of proteins per output file chunk
        padding_idx: Index to use for padding
        trim_start: Number of residues to trim from start of sequence
        trim_end: Number of residues to trim from end of sequence
    """
    # First get all tokens
    output_prefix = Path(output_prefix)
    print("Tokenizing sequences...")
    tokenized_sequences, masked_indices = tokenize_sequences(model, sequence_file, max_length, overlap, trim_start, trim_end, mask_prob)
    print(f"Tokenized {len(tokenized_sequences)} sequences")
    
    # Save tokens file
    tokens_file = output_prefix.parent / f"{output_prefix.name}_tokens.pt"
    torch.save({k: v[trim_start:-trim_end] for k, v in tokenized_sequences.items()}, tokens_file)
    print(f"Saved tokens for {len(tokenized_sequences)} proteins")
    if mask_prob > 0:
        masked_file = output_prefix.parent / f"{output_prefix.name}_tokens.masked"
        with open(masked_file, "w") as f:
            for k, v in masked_indices.items():
                f.write(f"{k}\t{','.join(map(str, v.tolist()))}\n")
        print(f"Saved masked tokens for {len(masked_indices)} proteins")
    # Load ESM-2 model with 4-bit quantization
    if quantize:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        print("Not quantizing")
        bnb_config = None
    if "prot_t5" in model:
        model = T5ForConditionalGeneration.from_pretrained(
            model,
            torch_dtype="auto",
            quantization_config=bnb_config,
            device_map="cuda",
        ).encoder
    else:
        model = AutoModel.from_pretrained(
            model,
            torch_dtype="auto",
            quantization_config=bnb_config,
            add_pooling_layer=False,
            local_files_only=True,
            device_map="cuda",
        )
    model.eval()

    # Process sequences in batches
    protein_ids = list(tokenized_sequences.keys())
    embeddings_dict = {}
    chunk_count = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(protein_ids), batch_size), desc="Computing embeddings"):
            batch_ids = protein_ids[i:i+batch_size]
            batch_input_ids = [tokenized_sequences[pid] for pid in batch_ids]
            # Pad to max length in batch
            max_len = max(seq.size(0) for seq in batch_input_ids)
            batch_input_ids_padded = torch.stack([
                F.pad(seq, (0, max_len - seq.size(0)), value=padding_idx) for seq in batch_input_ids
            ])
            attention_mask = (batch_input_ids_padded != padding_idx).long()
            batch_input_ids_padded = batch_input_ids_padded.to(model.device)
            attention_mask = attention_mask.to(model.device)
            batch_embeddings = model(input_ids=batch_input_ids_padded, attention_mask=attention_mask).last_hidden_state.cpu()
            for j, pid in enumerate(batch_ids):
                seq_len = (attention_mask[j] != 0).sum().item()
                embeddings_dict[pid] = batch_embeddings[j, :seq_len].clone()
            # Save chunk if size threshold reached
            if len(embeddings_dict) >= chunk_size:
                chunk_count += 1
                embeddings_dict = {k: v[trim_start:-trim_end] for k, v in embeddings_dict.items()}
                assert all(len(v) == len(tokenized_sequences[k][trim_start:-trim_end]) for k, v in embeddings_dict.items()), f"Sequence length mismatch for {chunk_count}"
                chunk_file = output_prefix.parent / f"{output_prefix.name}_embeddings_{chunk_count}.pt"
                torch.save(embeddings_dict, chunk_file)
                print(f"Saved chunk {chunk_count} with {len(embeddings_dict)} proteins")
                embeddings_dict = {}
                torch.cuda.empty_cache()

        # Save final chunk
        if embeddings_dict:
            chunk_count += 1
            embeddings_dict = {k: v[trim_start:-trim_end] for k, v in embeddings_dict.items()}
            chunk_file = output_prefix.parent / f"{output_prefix.name}_embeddings_{chunk_count}.pt"
            torch.save(embeddings_dict, chunk_file)
            print(f"Saved final chunk {chunk_count} with {len(embeddings_dict)} proteins")



def main():
    parser = argparse.ArgumentParser(description='Generate ESM2 embeddings from a FASTA file')
    parser.add_argument('--model', type=str, default="facebook/esm2_t33_650M_UR50D", help='Model to use')
    parser.add_argument('--sequence_file', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--output_prefix', type=Path, required=True, help='Output prefix for embeddings and tokens')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--padding_idx', type=int, default=1, help='Padding index')
    parser.add_argument('--trim_start', type=int, default=1, help='Trim start of sequence')
    parser.add_argument('--trim_end', type=int, default=1, help='Trim end of sequence')
    parser.add_argument('--mask_prob', type=float, default=0, help='Mask probability')
    parser.add_argument('--chunk_size', type=int, default=2500, help='Chunk size for saving embeddings')
    parser.add_argument('--overlap', type=int, default=128, help='Overlap size for sequence chunks')
    parser.add_argument('--quantize', action='store_true', help='Whether to quantize the embeddings')
    args = parser.parse_args()
    embed_sequences(model=args.model, 
                    sequence_file=args.sequence_file, 
                    output_prefix=args.output_prefix, 
                    batch_size=args.batch_size, 
                    chunk_size=args.chunk_size, 
                    padding_idx=args.padding_idx, 
                    max_length=args.max_length, 
                    overlap=args.overlap, 
                    trim_start=args.trim_start, 
                    trim_end=args.trim_end,
                    mask_prob=args.mask_prob,
                    quantize=args.quantize)

if __name__ == "__main__":
    main()
