import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.nn import functional as F


DEFAULT_CHARACTERS = "ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy"


class Tea(nn.Module, PyTorchModelHubMixin, repo_url="tea", license="mit"):
    """
    The Embedded Alphabet (tea) model for converting input pLMs embeddings into tea sequences.

    This model consists of two linear layers with dropout and normalization. It provides methods 
    to compute Shannon entropy over the output distribution and to convert model outputs into 
    character sequences.

    Args:
        representation_size (int): Dimensionality of input representations.
        hidden_size (int): Hidden size for first linear transformation.
        codebook_size (int): Number of unique tokens (characters) in the alphabet.
        dropout_prob (float): Dropout probability for regularization.
        ignore_token_ids (list[int]): Token ids to ignore when constructing sequences.
    """

    def __init__(
        self,
        representation_size: int = 1280,
        hidden_size: int = 1280,
        codebook_size: int = 20,
        dropout_prob: float = 0.1,
        ignore_token_ids: list[int] = [0, 1, 2],
    ):
        super().__init__()
        self.representation_size = representation_size
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.ignore_token_ids = ignore_token_ids

        self.dense = nn.Linear(representation_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, codebook_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.eps = 1e-8

        self.characters = list(DEFAULT_CHARACTERS[:codebook_size])

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def compute_shannon_entropy(self, logits):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1) / torch.log(
            torch.tensor(self.codebook_size, dtype=probs.dtype, device=probs.device)
        )
        return entropy

    def to_sequences(
        self,
        input_ids,
        embeddings,
        attention_mask=None,
        logits=None,
        return_logits=False,
    ):
        if logits is None:
            logits = self(embeddings)
        ignore_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in self.ignore_token_ids:
            ignore_mask &= input_ids != token_id
        predicted_indices = torch.argmax(logits, dim=-1)

        sequences = []
        logits_list = []

        for seq_idx, seq_logits, mask in zip(predicted_indices, logits, ignore_mask):
            filtered_indices = seq_idx[mask]
            sequence = "".join(self.characters[idx.item()] for idx in filtered_indices)
            sequences.append(sequence)
            if return_logits:
                logits_list.append(seq_logits[mask])

        if not return_logits:
            return sequences

        return {"sequences": sequences, "logits": logits_list}
