# 3rd Party
import torch
from torch.utils.tensorboard import SummaryWriter
# Local
from lm import LanguageModel, ReviewDataset
# Python STL
import os
from typing import List, Dict
import time
import pdb


def train(model         : torch.nn.Module,
        optimizer       : torch.optim.Optimizer,
        vocab           : List[str],
        train_files     : List[str],
        val_files       : List[str]             = [],
        batch_size      : int                   = 32,
        batches_per_era : int                   = 1000,
        max_train_eras  : int                   = 5,
        max_val_batches : int                   = 100,
        patience        : int                   = 5,
        save_dir        : str                   = None,
        log_dir         : str                   = None) -> Dict[str, object]:

    timestamp = f"{int(time.time())}"

    if not log_dir:
        log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    if not save_dir:
        save_dir = os.path.join("ckpts", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    train_set    = ReviewDataset(train_files, vocab, repeat = True)
    val_set      = ReviewDataset(val_files  , vocab, repeat = False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # Metrics
    criterion    = torch.nn.CrossEntropyLoss(reduction="sum")

    checkpoint = {
        "model"             : model.state_dict(),
        "optimizer"         : optimizer.state_dict(),
        "era"               : torch.tensor(1),
        "waited"            : torch.tensor(0),
        "best_avg_val_loss" : torch.tensor(float("inf"))
    }

    era_index         : torch.Tensor = checkpoint["era"]
    waited            : torch.Tensor = checkpoint["waited"]
    best_avg_val_loss : torch.Tensor = checkpoint["best_avg_val_loss"]

    era_loss    = 0.
    era_samples = 0
    for [i, (x1,x2,x3)] in enumerate(train_loader, start=1):
        loss         = criterion(model(torch.stack([x1,x2], dim=-1)), x3)
        era_loss    += loss.item()
        era_samples += len(x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % batches_per_era == 0:

            writer.add_scalar("Loss/train", era_loss/era_samples, era_index)

            val_loss = 0
            val_samples = 0
            for [j, (x1,x2,x3)] in enumerate(torch.utils.data.DataLoader(val_set, batch_size=batch_size), start=1):
                val_loss    += criterion(model(torch.stack([x1,x2], dim=-1)), x3).item()
                val_samples += len(x1)
                if j >= max_val_batches: break
            val_average = val_loss/val_samples
            writer.add_scalar("Loss/validation", val_average, era_index)

            # Early stopping
            if val_average <= best_avg_val_loss:
                best_avg_val_loss.data = torch.tensor(val_average)
                print(f"New best validation loss: {val_average}")
                waited.data  = torch.tensor(0)
                torch.save(checkpoint, os.path.join(save_dir, "best.pt"))
            else:
                waited += 1
                if waited >= patience:
                    break
            era_index   += 1
            era_loss     = 0.
            era_samples  = 0
        
    torch.save(checkpoint, os.path.join(save_dir, f"{era_index}_eras.pt"))
    return checkpoint


if __name__ == "__main__":
    vocab_path     = "data/vocab.txt"
    in_tokens      = 2
    embedding_size = 128
    with open(vocab_path) as r: vocab = list(map(lambda l: l.strip(), r.readlines()))
    assert len(vocab) == len(set(vocab))
    vocab_size = len(vocab) + 1

    model      = LanguageModel(in_tokens, vocab_size, embedding_size)
    optimizer  = torch.optim.Adam(model.parameters(), lr = 0.01)

    train(model, optimizer, vocab, ["data/parted/0.txt"], ["data/parted/1.txt"],
        batch_size=32,
        max_train_eras=100,
        batches_per_era=100,
        max_val_batches=10)

