"""
Custom data loader for WikiText2 dataset from torchtext.
For use with the official PyTorch Transformer tutorial.
https://pytorch.org/tutorials/beginner/transformer_tutorial.html

Provides processing of bare WikiText2 data into batches for
training, validating, and testing the Transformer.
"""

# from torch.utils.data import dataset
# from torchtext.datasets import WikiText2

def tokenize(seq):
    """
    Implements the "basic_english" functionality of
    torchtext.data.utils.get_tokenizer applied to a
    text sequence seq.
    """
    return (seq.lower()).split(' ')




