# The Embedding Alphabet (TEA)

![Model Architecture](Model_Architecture.png)

This repository contains the code accompanying our pre-print (link coming soon).

## Installation

```bash
pip install git+https://github.com/sliM-Lp/tea.git
```

## Sequence Conversion with TEA

The `tea_convert` command takes protein sequences from a FASTA file and generates new tea-FASTA. It supports confidence-based sequence output where low-confidence positions are displayed in lowercase, and has options for saving logits and entropy. If `--save_avg_entropy` is set, the FASTA identifiers will contain the average entropy of the sequence in the format `<key>|H=<avg_entropy>`.

```bash
usage: tea_convert [-h] -f FASTA_FILE -o OUTPUT_FILE [-s] [-e] [-r] [-l] [-t ENTROPY_THRESHOLD]

options:
  -h, --help            show this help message and exit
  -f FASTA_FILE, --fasta_file FASTA_FILE
                        Input FASTA file containing protein amino acid sequences
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output FASTA file for generated tea sequences
  -l, --save_logits     Save per-residue logits to .pt file
  -H, --save_avg_entropy
                        Save average entropy values in FASTA identifiers
  -r, --save_residue_entropy
                        Save per-residue entropy values to .pt file
  -c, --lowercase_entropy
                        Save residues with entropy > threshold in lowercase
  -t ENTROPY_THRESHOLD, --entropy_threshold ENTROPY_THRESHOLD
                        Entropy threshold for lowercase conversion
```

### Using the huggingface model

```python
from tea.model import Tea
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
import torch
import re

tea = Tea.from_pretrained("PickyBinders/tea")
device = next(tea.parameters()).device
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
bnb_config = BitsAndBytesConfig(load_in_4bit=True) if torch.cuda.is_available() else None
esm2 = AutoModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D",
        torch_dtype="auto",
        quantization_config=bnb_config,
        add_pooling_layer=False,
    ).to(device)
esm2.eval()
sequence_examples = ["PRTEINO", "SEQWENCE"]
sequence_examples = [" ".join(list(re.sub(r"[UZOBJ]", "X", sequence))) for sequence in sequence_examples]
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)
with torch.no_grad():
    x = esm2(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state.to(device)
    results = tea.to_sequences(embeddings=x, input_ids=input_ids, return_avg_entropy=True, return_logits=False, return_residue_entropy=False)
results
```

## Using tea sequences with MMseqs2

The `matcha.out` substitution matrix is included with the tea package. You can get its path programmatically:

```python
from tea import get_matrix_path
matcha_path = get_matrix_path()
print(f"Matrix path: {matcha_path}")
```

Then use it with MMseqs2:

```bash
mmseqs easy-search tea_query.fasta tea_target.fasta results.m8 tmp/ \
    --comp-bias-corr 0 \
    --mask 0 \
    --gap-open 18 \
    --gap-extend 3 \
    --sub-mat /path/to/matcha.out \
    --seed-sub-mat /path/to/matcha.out \
    --exact-kmer-matching 1
```
