import torch
import numpy as np
from scipy import spatial

def compute_similarity_matrix(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: multiplier for the distances in the exponential kernel
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    
    sm = compute_similarity_matrix_plain(embedding1, embedding2, l=l, p=p)
    columns_avg = torch.sum(sm,0)/sm.shape[0]
    rows_avg = torch.sum(sm,1)/sm.shape[1]
    
    columns_std = torch.std(sm,0)
    rows_std = torch.std(sm,1)

    z_rows = (sm-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (sm-columns_avg)/columns_std
    
    return (z_rows+z_columns)/2


def compute_similarity_matrix_plain(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the plain
        similarity matrix.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: multiplier for the distances in the exponential kernel
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """

    return torch.exp(-l*torch.cdist(embedding1, embedding2, p=p))



def compute_cosine_similarity_matrix(embedding1, embedding2):
    """Compute cosine similarities with row and column Z-score enhancement.

        For direct cosine scores, use compute_cosine_similarity_matrix_plain.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
    """
    
    sm = compute_cosine_similarity_matrix_plain(embedding1, embedding2)
    columns_avg = torch.sum(sm,0)/sm.shape[0]
    rows_avg = torch.sum(sm,1)/sm.shape[1]
    
    columns_std = torch.std(sm,0)
    rows_std = torch.std(sm,1)

    z_rows = (sm-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (sm-columns_avg)/columns_std
    
    return (z_rows+z_columns)/2


def compute_cosine_similarity_matrix_plain(embedding1, embedding2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the cosine
        similarity matrix.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
    """
    
    return torch.tensor(1-spatial.distance.cdist(embedding1.cpu().numpy(), embedding2.cpu().numpy(), 'cosine'))

