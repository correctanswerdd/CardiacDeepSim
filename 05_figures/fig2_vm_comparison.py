"""Fig 2: normalised Vm of cell (12, 20) over 2000 steps: golden standard, LSTM-assisted simulation, Neural ODE.

Both models start at step 1 of the test set. The Neural ODE is solved
for 12 cycles and its first 2000 steps are shown.

    python3 05_figures/fig2_vm_comparison.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _style  # noqa: E402,F401
from _style import plt  # noqa: E402
from common import load_lstm, load_node, lstm_simulate, node_simulate, out_dir, remap, test_data  # noqa: E402

s, cell, length = 1, 12 * 24 + 20, 2000
data = test_data()
lstm = lstm_simulate(load_lstm(), data, s, length)[:, cell, 0]
node = node_simulate(load_node(), data, s, n_cycles=12)[:length, cell, 0]
truth = data[s:s + length, cell, 0]

d = out_dir("05_figures")
np.savez_compressed(os.path.join(d, "fig2_vm_comparison.npz"), start=s, cell=cell, lstm=lstm, node=node, truth=truth)

plt.figure(figsize=(20, 5))
x_ax = range(length)
plt.plot(x_ax, remap(lstm, -1, 1), label="LSTM-assisted Simulation", color="cornflowerblue", linewidth=5)
plt.plot(x_ax, remap(node, -1, 1), label="Neural ODE Simulation", color="lightcoral", linewidth=5)
plt.plot(x_ax, remap(truth, -1, 1), label="Golden Standard", color="midnightblue", linestyle="dashed", linewidth=3)
plt.xlabel("steps", fontsize=24)
plt.ylabel("Vm", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(bbox_to_anchor=(1., 1.), fontsize=28)
path = os.path.join(d, "fig2_vm_comparison.png")
plt.savefig(path, dpi=500, bbox_inches="tight")
print("wrote outputs/05_figures/fig2_vm_comparison.png")
