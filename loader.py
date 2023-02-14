"""
Custom data loader/batcher for WikiText2 dataset from torchtext.
Largely taken from the official PyTorch Transformer tutorial.
https://pytorch.org/tutorials/beginner/transformer_tutorial.html

Provides processing of bare WikiText2 data into batches for
training, validating, and testing the Transformer.
"""

import re
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import Tensor
from typing import Tuple

class Tokenizer:
    def __init__(self) -> None:
        regexes = [r"\'", r"\"", r"\.", r"<br \/>", r",", 
            r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
        switches = [" '  ", "", " . ", " ", " , ", 
            " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
        self.reg_map = [(re.compile(expr),rep) for expr,rep in zip(regexes,switches)]

    def tokenize(self,seq):
        """
        Implements the "basic_english" functionality of
        torchtext.data.utils.get_tokenizer applied to a
        text sequence seq.

        Regular expressions to normalize string and then
        split on any whitespace.
        Strategy is taken from get_tokenizer source.
        https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
        """
        seq = seq.lower()
        for regex,switch in self.reg_map:
            seq = regex.sub(switch, seq)
        return seq.split()


class Processor:
    def __init__(self) -> None:
        print("Loading datasets...")
        self.train_iter, self.val_iter, self.test_iter = WikiText2()
        self.token = Tokenizer()
        self.tokenizer = self.token.tokenize
        self.vocab = build_vocab_from_iterator(
            map(self.tokenizer, self.train_iter), specials=['<unk>']
            )
        self.vocab.set_default_index(self.vocab['<unk>'])
        print("Reloading datasets...")
        self.train_iter = WikiText2(split='train') # Reload iterator after vocab
        
    def data_process(self, raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def generate(self):
        return (self.data_process(self.train_iter),
                self.data_process(self.val_iter),
                self.data_process(self.test_iter)
                )


class Batcher:
    def __init__(self) -> None:
        self.proc = Processor()
        self.train, self.val, self.test = self.proc.generate()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def batchify(self, data: Tensor, bsz: int) -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(self.device)

    def generate(self, batch_size, eval_batch_size):
        print("Batchifying datasets...")
        train_data = self.batchify(self.train, batch_size)  # shape [seq_len, batch_size]
        val_data = self.batchify(self.val, eval_batch_size)
        test_data = self.batchify(self.test, eval_batch_size)
        return train_data, val_data, test_data


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target
