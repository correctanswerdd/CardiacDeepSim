"""Step 04: Table 1 (MSE of Vm, Cai, CaSR, Cass for the LSTM-assisted simulation and the Neural ODE).

Both models start from the same point of the test set and simulate one cycle (500 steps); Eq. (9) is the mean
squared error over these 500 steps and all 576 cells, each attribute rescaled to [-1, 1].

The start point is step 2067 of the test set. The Neural ODE draws z0 from q(z0) with a fixed seed
(common.NODE_SEED).

    python3 04_evaluate/table1.py [--checkpoints checkpoints|outputs/03_train] [--start 2067]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (ATTRS, CYCLE_LEN, NODE_SEED, ROOT, load_lstm, load_node, lstm_simulate, mse_per_attr,  # noqa: E402
                    node_simulate, out_dir, test_data, write_json)

TABLE1_START = 2067

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", default="checkpoints", help="folder with LSTMbest.pt / NeuralODE.pt "
                    "(or LSTM.pt / NeuralODE.pt from step 03)")
parser.add_argument("--start", type=int, default=TABLE1_START)
args = parser.parse_args()

ck = os.path.join(ROOT, args.checkpoints)
lstm_ck = os.path.join(ck, "LSTMbest.pt") if os.path.isfile(os.path.join(ck, "LSTMbest.pt")) else os.path.join(ck, "LSTM.pt")
node_ck = os.path.join(ck, "NeuralODE.pt")
data = test_data()
s = args.start

mse = {
    "Neural ODE Simulation": mse_per_attr(node_simulate(load_node(node_ck), data, s, n_cycles=2), data, s),
    "LSTM-assisted Simulation": mse_per_attr(lstm_simulate(load_lstm(lstm_ck), data, s, CYCLE_LEN), data, s),
}

rows = ["| | " + " | ".join(ATTRS) + " |", "|---|" + "---|" * len(ATTRS)]
for model, values in mse.items():
    rows.append(f"| {model} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
print("\n".join(rows))

d = out_dir("04_evaluate")
write_json(os.path.join(d, "table1.json"), {
    "start_in_test_set": s, "steps": CYCLE_LEN, "node_seed": NODE_SEED,
    "checkpoints": {"lstm": os.path.relpath(lstm_ck, ROOT), "node": os.path.relpath(node_ck, ROOT)},
    "attributes": ATTRS, "mse": mse})
with open(os.path.join(d, "table1.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
