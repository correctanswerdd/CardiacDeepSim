"""Step 03a: train the LSTM-assisted simulation (Algorithm 1).

Hyper-parameters of Section 4.3: Adam, learning rate 1e-3, batch size 16, 2 LSTM layers (hidden size 64,
dropout 0.2), 3 epochs; training sequences of 1000 steps sampled from the first 80 % of the data, z-scored.
After every batch the model is rolled out for 500 steps on the following data and that error is logged;
the checkpoint with the lowest such validation error is kept as outputs/03_train/LSTM.pt.

    python3 03_train/train_lstm.py [--epochs 3] [--seed 0]
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import LSTM, NUM_LAYERS, load_dataset, out_dir, preprocess, split, write_json  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-batches", type=int, default=None, help="stop each epoch after this many batches (smoke test)")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

learning_rate, batch_size, seq_len, val_seq_len = 0.001, 16, 1000, 500


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, val_seq_len):
        self.seq_len, self.val_seq_len = seq_len, val_seq_len
        self.data = torch.from_numpy(preprocess(data)).float()  # shape (time, cells, attributes)

    def __len__(self):
        return self.data.shape[0] - self.val_seq_len - self.seq_len - 1

    def __getitem__(self, i):
        s, v = self.seq_len, self.val_seq_len
        return (self.data[i:i + s], self.data[i + 1:i + 1 + s],
                self.data[i + 1 + s:i + 1 + s + v], self.data[i + 2 + s:i + 2 + s + v])


train_data, _ = split(load_dataset())
train_loader = DataLoader(TimeSeriesDataset(train_data.copy(), seq_len, val_seq_len), batch_size=batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(args.seed))
print("training sequences", len(train_loader.dataset), "batches per epoch", len(train_loader))

model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)
criterion = nn.MSELoss()


def validation(b_, data_val, future):
    model.reset(b_)
    inpt = model(data_val)[:, -1].unsqueeze(1)
    preds = []
    for _ in range(future):
        inpt = model(inpt)
        preds.append(inpt)
    return torch.cat(preds, axis=1)


d = out_dir("03_train")
log, best_val, t0 = [], float("inf"), time.time()
for epoch in range(args.epochs):
    for bi, (data, target, data_val, target_val) in enumerate(train_loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        model.train()
        b_ = data.size(0)
        data, target = data.to(device), target.to(device)
        data_val, target_val = data_val.to(device), target_val.to(device)
        optimizer.zero_grad()
        model.reset(b_)
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(validation(b_, data_val, val_seq_len), target_val).item()
        log.append((epoch, bi, loss.item(), valid_loss))
        print(f"Epoch {epoch + 1}, Batch {bi}: Loss {loss.item()}, Validation {valid_loss}", flush=True)
        if valid_loss < best_val:
            best_val = valid_loss
            torch.save(model.state_dict(), os.path.join(d, "LSTM.pt"))
    scheduler.step()
    torch.save(model.state_dict(), os.path.join(d, f"LSTM_epoch{epoch}.pt"))

np.savetxt(os.path.join(d, "LSTM_loss.csv"), np.array(log), delimiter=",", header="epoch,batch,train_mse,val_mse")
write_json(os.path.join(d, "LSTM.json"), {"seed": args.seed, "epochs": args.epochs, "best_val_mse": best_val,
                                          "seconds": time.time() - t0, "num_layers": NUM_LAYERS})
