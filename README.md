# The Embedded Alphabet (TEA)

![Model Architecture](Model_Architecture.png)

This repository contains the code accompanying our pre-print: [Rewriting protein alphabets with language models](https://doi.org/10.1101/2025.11.27.690975). A web server with TEA converted datasets and search capabilities is available [here](https://pickybinders.org/tea).

## Installation

```bash
python -m pip install git+https://github.com/PickyBinders/tea.git
```
* Tested on Python 3.11, 3.12 and 3.13
* Typical installation time: 2min

## Sequence Conversion with TEA

The `tea_convert` command takes protein sequences from a FASTA file and generates a TEA FASTA. It uses the ESM2-650M model and the TEA checkpoint from the Hugging Face `dev` revision. Length-based batching is automatic, and conversion time is reported after completion. Optionally, residues with low raw-logit spread can be written in lowercase, and mean logit spread (`TLS`) and scale-free certainty (`TCP`) can be added to FASTA headers.

```bash
tea_convert --fasta proteins.fasta --output proteins.tea.fasta \
  --lowercase-logit-spread-below 3.5 --confidence-headers
```

```text
usage: tea_convert [-h] --fasta FASTA --output OUTPUT
                   [--lowercase-logit-spread-below LOWERCASE_LOGIT_SPREAD_BELOW]
                   [--confidence-headers]

options:
  -h, --help            show this help message and exit
  --fasta FASTA, -f FASTA
  --output OUTPUT, -o OUTPUT
  --lowercase-logit-spread-below LOWERCASE_LOGIT_SPREAD_BELOW
                        Lowercase residues with raw-logit spread below this value
  --confidence-headers  Append mean logit spread (TLS) and scale-free certainty (TCP)
```

### Using the huggingface model

```python
from tea.model import Tea
from tea.convert import _enable_fp32_rotary
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
import torch
import re

tea = Tea.from_pretrained("PickyBinders/tea", revision="dev").to("cuda")
tea.eval()
device = next(tea.parameters()).device
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/esm2_t33_650M_UR50D", revision="08e4846e537177426273712802403f7ba8261b6c"
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16
)
esm2 = AutoModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D",
        revision="08e4846e537177426273712802403f7ba8261b6c",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={"": str(device)},
        add_pooling_layer=False,
    )
esm2.eval()
_enable_fp32_rotary(esm2)
sequence_examples = ["PRTEINO", "SEQWENCE"]
sequence_examples = [" ".join(list(re.sub(r"[UZOBJ]", "X", sequence))) for sequence in sequence_examples]
ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)
with torch.no_grad():
    x = esm2(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state
    results = tea.to_sequences(embeddings=x.half().float(), input_ids=input_ids)
results
```

## Search with TEA against Many

In order to perform fast sequence searches and generate alignments, we recommend checking out STEAM. This tool is designed to leverage both TEA representations and standard amino acid information, allowing you to execute comprehensive dual-character sequence screening against large datasets. You can find the repository and usage instructions at [github.com/PickyBinders/steam](https://github.com/PickyBinders/steam).
