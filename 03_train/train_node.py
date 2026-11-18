"""Step 03b: train the Neural ODE benchmark (Section 4.2).

Latent ODE (Chen et al. 2018): recognition RNN (hidden 20) -> q(z0) (latent 50) -> ODE function (hidden 50,
torchdiffeq odeint) -> decoder (hidden 50). Training windows of 1000 steps from the first 80 % of the data,
z-scored and then each attribute rescaled to [-1, 1] inside the window; ELBO with observation noise 0.005;
Adam, learning rate 0.01, batch size 16, 2 passes over the data. The model is saved whenever the running average
of the loss improves; the last such save is outputs/03_train/NeuralODE.pt.

    python3 03_train/train_node.py [--epochs 2] [--seed 0]
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (LATENT_DIM, N_ATTR, N_CELLS, NODE_SEQ_LEN, Decoder, LatentODEfunc, RecognitionRNN,  # noqa: E402
                    load_dataset, out_dir, preprocess, remap_attrs, split, write_json)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-batches", type=int, default=None, help="stop each epoch after this many batches (smoke test)")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

batch_size, noise_std = 16, 0.005


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = preprocess(data)

    def __len__(self):
        return self.data.shape[0] - self.seq_len - 1

    def __getitem__(self, i):
        return torch.from_numpy(remap_attrs(self.data[i:i + self.seq_len])).float()


class RunningAverageMeter:
    def __init__(self, momentum=0.99):
        self.momentum, self.val, self.avg = momentum, None, 0

    def update(self, val):
        self.avg = val if self.val is None else self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    return -.5 * (logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1, v2 = torch.exp(lv1), torch.exp(lv2)
    return lv2 / 2. - lv1 / 2. + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5


train_data, _ = split(load_dataset())
train_loader = DataLoader(TimeSeriesDataset(train_data.copy(), NODE_SEQ_LEN), batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed))
samp_ts = torch.from_numpy(np.linspace(0, 1, num=NODE_SEQ_LEN)).float().to(device)

func, rec, dec = LatentODEfunc().to(device), RecognitionRNN(nbatch=batch_size).to(device), Decoder().to(device)
optimizer = optim.Adam(list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()), lr=0.01)
loss_meter = RunningAverageMeter()

d = out_dir("03_train")
log, min_loss, t0 = [], float("inf"), time.time()
for itr in range(1, args.epochs + 1):
    for bi, samp_trajs in enumerate(train_loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        b_, seq_l = samp_trajs.size(0), samp_trajs.size(1)
        samp_trajs = samp_trajs.to(device)
        optimizer.zero_grad()
        h = rec.initHidden(batch_size=b_).to(device)
        for t in reversed(range(seq_l)):
            out, h = rec(samp_trajs[:, t].reshape(b_, N_CELLS * N_ATTR), h)
        qz0_mean, qz0_logvar = out[:, :LATENT_DIM], out[:, LATENT_DIM:]
        z0 = torch.randn(qz0_mean.size(), device=device) * torch.exp(.5 * qz0_logvar) + qz0_mean
        pred_x = dec(odeint(func, z0, samp_ts).permute(1, 0, 2)).view(b_, seq_l, N_CELLS, N_ATTR)
        noise_logvar = 2. * torch.log(torch.zeros(pred_x.size(), device=device) + noise_std)
        # summed over the whole batch (four .sum(-1) on a 4-D tensor)
        logpx = log_normal_pdf(samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1).sum(-1).sum(-1)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar, torch.zeros_like(z0), torch.zeros_like(z0)).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        log.append((itr, bi, loss.item(), loss_meter.avg))
        print(f"Epoch {itr}, Batch {bi}: running avg loss {loss_meter.avg}", flush=True)
        if loss_meter.avg < min_loss:
            min_loss = loss_meter.avg
            torch.save({"func_dict": func.state_dict(), "rec_dict": rec.state_dict(), "dec_dict": dec.state_dict()},
                       os.path.join(d, "NeuralODE.pt"))

np.savetxt(os.path.join(d, "NeuralODE_loss.csv"), np.array(log), delimiter=",",
           header="epoch,batch,loss,running_avg_loss")
write_json(os.path.join(d, "NeuralODE.json"), {"seed": args.seed, "epochs": args.epochs,
                                               "min_running_avg_loss": min_loss, "seconds": time.time() - t0})
