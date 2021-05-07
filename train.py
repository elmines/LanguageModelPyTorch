# 3rd Party
import torch
# Local
from lm import LanguageModel, ReviewDataset
# Python STL
import pdb
from typing import List


def train(model     : torch.nn.Module,
        vocab       : List[str],
        train_files : List[str],
        val_files   : List[str] = []):

    batch_size = 32
    max_epochs = 5

    train_set = ReviewDataset(train_files, vocab)
    val_files = ReviewDataset(val_files, vocab)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    for epoch in range(2):
        for (i,sample) in enumerate(train_loader):
            print(f"{epoch},{i}")


if __name__ == "__main__":

    vocab_path     = "data/vocab.txt"
    in_tokens      = 2
    embedding_size = 128
    with open(vocab_path) as r: vocab = list(map(lambda l: l.strip(), r.readlines()))
    assert len(vocab) == len(set(vocab))
    vocab_size = len(vocab) + 1
    model = LanguageModel(in_tokens, vocab_size, embedding_size)
    train(model, vocab, ["data/parted/0.txt"], ["data/parted/1.txt"])

