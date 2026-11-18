"""Fig 3: LSTM-assisted simulation of Vm, Cai, CaSR and Cass for four cells, each from its own start step.

(cell, start in the test set) as in lstm_neuralode_comparison.ipynb: (6, 21) at 1935, (12, 20) at 3657,
(10, 14) at 3032, (1, 7) at 2067. Row 2 shows the cell's Vm over the first 6000 test steps with the start marked.

    python3 05_figures/fig3_long_term.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _style  # noqa: E402,F401
from _style import plt  # noqa: E402
from common import ATTRS, load_lstm, lstm_simulate, out_dir, remap, test_data  # noqa: E402

show_data = [(6 * 24 + 21, 1935), (12 * 24 + 20, 3657), (10 * 24 + 14, 3032), (1 * 24 + 7, 2067)]
vis = 2000
data = test_data()
model = load_lstm()

fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(10, 10))
saved = {}
for col, (cell, s) in enumerate(show_data):
    cell_x, cell_y = cell // 24, cell % 24
    X = lstm_simulate(model, data, s, vis)
    saved[f"col{col}_lstm"] = X[:, cell, :]
    saved[f"col{col}_truth"] = data[s:s + vis, cell, :]
    for row in range(6):
        ax = axs[row, col]
        if row == 0:
            ax.set_title(f"cell_({cell_x}, {cell_y}) location")
            ax.add_artist(plt.Rectangle((0, 0), 24, 24, color="lightgray"))
            ax.set_xlim(0, 24)
            ax.set_ylim(0, 24)
            ax.set_aspect("equal")
            ax.xaxis.set_visible(False)
            ax.invert_yaxis()
            ax.plot(cell_y, cell_x, color="lightcoral", marker="o", markersize=10)
            continue
        if row == 1:
            ax.plot(range(6000), data[:6000, cell, 0], color="darkgrey")
            ax.axvline(x=s, color="lightcoral", linestyle="--")
            ax.yaxis.set_visible(False)
            continue
        k = row - 2
        ax.set(ylabel=ATTRS[k])
        if row == 5:
            ax.set(xlabel="steps")
        ax.grid(color="grey", linestyle="--", linewidth=0.5)
        ax.plot(range(vis), remap(X[:vis, cell, k], -1, 1), label="LSTM-assisted Simulation",
                color="cornflowerblue", linewidth=3)
        ax.plot(range(vis), remap(data[s:s + vis, cell, k], -1, 1), label="Golden Standard",
                color="midnightblue", linestyle="dashed")

for ax in axs.flat:
    ax.label_outer()
fig.tight_layout()
axs[3, 3].legend(bbox_to_anchor=(1., 1.))

d = out_dir("05_figures")
np.savez_compressed(os.path.join(d, "fig3_long_term.npz"), cells_starts=np.array(show_data), **saved)
plt.savefig(os.path.join(d, "fig3_long_term.png"), dpi=500, bbox_inches="tight")
print("wrote outputs/05_figures/fig3_long_term.png")
