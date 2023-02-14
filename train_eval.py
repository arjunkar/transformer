"""
Training and evaluation architecture taken from WikiText2 tutorial.
"""

import time
import math
import torch
from torch import nn
from torch import Tensor
from loader import get_batch
import os
from tempfile import TemporaryDirectory


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class TrainEval:
    def __init__(self, model, criterion, optimizer, scheduler, 
                train_data, val_data, bptt, device, prof=None) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.bptt = bptt
        self.prof = prof

    def train(self, epoch) -> None:
        self.model.train()  # turn on train mode
        if self.prof is not None:
            self.prof.start()
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(self.bptt).to(self.device)

        num_batches = len(self.train_data) // self.bptt
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = get_batch(self.train_data, i, self.bptt)
            seq_len = data.size(0)
            if seq_len != self.bptt:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = self.model(data, src_mask)
            ntokens = output.size(dim=-1)
            loss = self.criterion(output.view(-1, ntokens), targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            if self.prof is not None:
                self.prof.step()
            print("Gradient step "+str(i)+" completed...")
            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()
        if self.prof is not None:
            self.prof.stop()

    def train_model(self, epochs):
        best_val_loss = float('inf')
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                self.train(epoch)
                val_loss = self.evaluate(self.val_data)
                val_ppl = math.exp(val_loss)
                elapsed = time.time() - epoch_start_time
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                    f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
                print('-' * 89)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), best_model_params_path)

                self.scheduler.step()
            self.model.load_state_dict(torch.load(best_model_params_path)) # load best model states

    def evaluate(self, eval_data) -> float:
        self.model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(self.bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, self.bptt):
                data, targets = get_batch(eval_data, i)
                seq_len = data.size(0)
                if seq_len != self.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = self.model(data, src_mask)
                ntokens = output.size(dim=-1)
                output_flat = output.view(-1, ntokens)
                total_loss += seq_len * self.criterion(output_flat, targets).item()
        return total_loss / (len(eval_data) - 1)
