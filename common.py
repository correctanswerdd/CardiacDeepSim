"""Shared code for steps 02-05: paths, data loading, normalisation, the two models and their rollouts."""
import hashlib
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")
PUBLISHED = os.environ.get("PUBLISHED_DIR_IN_CONTAINER", os.path.join(ROOT, "published"))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")

# The paper's data set: 54801 time steps (dt = 1 ms) x 24 x 24 grid points x 4 attributes (Vm, Cai, CaSR, Cass).
# File name of the data set in published/.
PUBLISHED_DATA = "output0328-1.pickle"
PUBLISHED_DATA_SHA256 = "6319557b0fbcdf7990ac78393f43901a7034646e6597025ba7950391f4dee35a"
N_STEPS, GRID, N_ATTR = 54801, 24, 4
N_CELLS = GRID * GRID
ATTRS = ["Vm", "Cai", "CaSR", "Cass"]
CYCLE_LEN = 500  # steps per heartbeat (pacing cycle 500 ms)

# Expected split lengths and z-scored Vm range, used to check the data.
FINGERPRINT = {
    "n_train": 43840,
    "n_test": 10961,
    "full_vm_zscore_min": -1.0990786098543488,
    "full_vm_zscore_max": 3.2455547943257375,
    "test_vm_zscore_min": -1.0850763396420926,
    "test_vm_zscore_max": 3.24444338332082,
}

DEVICE = torch.device(os.environ.get("DEVICE", "cpu"))


def out_dir(step):
    d = os.path.join(OUTPUTS, step)
    os.makedirs(d, exist_ok=True)
    return d


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- data

DATASET = os.path.join(OUTPUTS, "02_make_data", "dataset.npy")


def load_dataset():
    """The (54801, 576, 4) float64 array written by step 02 (cell index = row * 24 + column)."""
    if not os.path.isfile(DATASET):
        raise SystemExit("outputs/02_make_data/dataset.npy is missing: run step 02 first (./run.sh make-data)")
    return np.load(DATASET)


def split(data):
    """80 % / 20 % split along time."""
    n_train = int(data.shape[0] * 0.8)
    return data[:n_train], data[n_train:]


def preprocess(a):
    """z-score each attribute over all steps and cells (in place; pass a copy)."""
    for k in range(N_ATTR):
        m, std = np.mean(a[:, :, k]), np.std(a[:, :, k])
        a[:, :, k] = (a[:, :, k] - m) / std
    return a


def remap(x, out_min, out_max):
    """Min-max rescale to [out_min, out_max]; used for the Neural ODE inputs, every MSE and every plot."""
    in_min, in_max = np.min(x), np.max(x)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def test_data():
    """The z-scored test set (10961, 576, 4) used by Table 1 and Fig 2-4."""
    _, test = split(load_dataset())
    return preprocess(test.copy())


# ---------------------------------------------------------------- LSTM-assisted simulation

HIDDEN_SIZE = 64
NUM_LAYERS = 2


class LSTM(nn.Module):
    def __init__(self, num_input_features=(N_CELLS, N_ATTR), hidden_layer_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        input_size = num_input_features[0] * num_input_features[1]
        self.num_layer = num_layers
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_layer_size, input_size)
        self.hidden_cell = None

    def forward(self, x):
        batch_size, seq_len, height, width = x.size()
        x = x.view(batch_size, seq_len, height * width)
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        preds = []
        for t in range(seq_len):
            pred = self.fc(lstm_out[:, t, :])
            preds.append(pred.view(batch_size, 1, height, width))
        return torch.cat(preds, axis=1)

    def reset(self, batch_size):
        z = torch.zeros(self.num_layer, batch_size, self.hidden_layer_size, device=next(self.parameters()).device)
        self.hidden_cell = (z, z.clone())


def load_lstm(path=None):
    model = LSTM()
    model.load_state_dict(torch.load(path or os.path.join(CHECKPOINTS, "LSTMbest.pt"), map_location="cpu"))
    return model.to(DEVICE).eval()


@torch.no_grad()
def lstm_simulate(model, data, start, future, warm_up=CYCLE_LEN):
    """Algorithm 2 (proposed_prediction in lstm_neuralode_comparison.ipynb).

    The hidden state is warmed up on the true steps start-1 .. start+warm_up-2, then the rollout starts again
    from the true step start-1 and feeds every prediction back as the next input. Output (future, 576, 4);
    row i is compared with data[start + i].
    """
    inputs = torch.from_numpy(data).float().to(DEVICE).unsqueeze(0)
    model.reset(1)
    model(inputs[:, start - 1:start + warm_up - 1])
    inpt = inputs[:, start - 1].unsqueeze(1)
    preds = []
    for _ in range(future):
        inpt = model(inpt)
        preds.append(inpt)
    return torch.cat(preds, axis=1)[0].cpu().numpy()


# ---------------------------------------------------------------- Neural ODE benchmark

LATENT_DIM, NHIDDEN, RNN_NHIDDEN = 50, 50, 20
OBS_DIM = N_CELLS * N_ATTR
NODE_SEQ_LEN = 1000
NODE_SEED = 0  # the reparameterisation noise of z0 is drawn with this seed, so every run gives the same numbers


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, nhidden=NHIDDEN):
        super().__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.fc3(self.elu(self.fc2(self.elu(self.fc1(x)))))


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, obs_dim=OBS_DIM, nhidden=RNN_NHIDDEN, nbatch=1):
        super().__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        h = torch.tanh(self.i2h(torch.cat((x, h), dim=1)))
        return self.h2o(h), h

    def initHidden(self, batch_size=None):
        return torch.zeros(self.nbatch if batch_size is None else batch_size, self.nhidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, obs_dim=OBS_DIM, nhidden=NHIDDEN):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        return self.fc2(self.relu(self.fc1(z)))


def load_node(path=None):
    ck = torch.load(path or os.path.join(CHECKPOINTS, "NeuralODE.pt"), map_location="cpu")
    func, rec, dec = LatentODEfunc(), RecognitionRNN(), Decoder()
    func.load_state_dict(ck["func_dict"])
    rec.load_state_dict(ck["rec_dict"])
    dec.load_state_dict(ck["dec_dict"])
    return tuple(m.to(DEVICE).eval() for m in (func, rec, dec))


def remap_attrs(x):
    x = x.copy()
    for k in range(N_ATTR):
        x[:, :, k] = remap(x[:, :, k], -1, 1)
    return x


@torch.no_grad()
def node_simulate(models, data, start, n_cycles, seed=NODE_SEED):
    """The Neural ODE rollout of lstm_neuralode_comparison.ipynb.

    The recognition RNN reads data[start : start+1000] (each attribute rescaled to [-1, 1]) backwards in time,
    z0 is drawn from q(z0) with a fixed seed, and the ODE is solved on n_cycles * 500 time points.
    Output (n_cycles * 500, 576, 4); row i is compared with data[start + i].
    """
    from torchdiffeq import odeint

    func, rec, dec = models
    inpts = torch.from_numpy(remap_attrs(data[start:start + NODE_SEQ_LEN])).float().unsqueeze(0).to(DEVICE)
    h = torch.zeros(1, RNN_NHIDDEN, device=DEVICE)
    for t in reversed(range(NODE_SEQ_LEN)):
        out, h = rec(inpts[:, t, :].reshape(1, OBS_DIM), h)
    qz0_mean, qz0_logvar = out[:, :LATENT_DIM], out[:, LATENT_DIM:]
    gen = torch.Generator().manual_seed(seed)
    epsilon = torch.randn(qz0_mean.size(), generator=gen).to(DEVICE)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    ts = torch.from_numpy(np.linspace(0, n_cycles / 2, num=CYCLE_LEN * n_cycles)).to(DEVICE)
    pred_z = odeint(func, z0, ts).permute(1, 0, 2)
    return dec(pred_z).view(-1, N_CELLS, N_ATTR).cpu().numpy()


def mse_per_attr(pred, data, start, length=CYCLE_LEN):
    """Eq. (9): mean squared error over one cycle (500 steps) and all 576 cells, each attribute rescaled to [-1, 1]
    separately for prediction and truth (error comparison cell of lstm_neuralode_comparison.ipynb)."""
    return [float(np.mean((remap(pred[:length, :, k], -1, 1) - remap(data[start:start + length, :, k], -1, 1)) ** 2))
            for k in range(N_ATTR)]
