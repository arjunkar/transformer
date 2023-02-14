"""
Structural code for training a transformer (encoder) model from
PyTorch's official examples.  Uses WikiText2 dataset.
"""


import math
import torch
from torch import nn

from layers import MyTransformer
from loader import Batcher
from train_eval import TrainEval


b = Batcher()
batch_size = 20
eval_batch_size = 10
train_data, val_data, test_data = b.generate(batch_size,eval_batch_size)
bptt = 35

print("Setting up transformer encoder model...")
ntokens = len(b.proc.vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = MyTransformer(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(b.device)

print("Model details:")
print("Vocab size: "+str(ntokens))

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('transformer/log/my_tr_run_2'),
        record_shapes=True,
        with_stack=True)

trainer = TrainEval(model, criterion, optimizer, scheduler, train_data, val_data, bptt,
                    b.device, prof)


print("Training loop beginning...")
trainer.train_model(epochs=3)
print("Training complete.")
print("Testing beginning....")
test_loss = trainer.evaluate(test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
