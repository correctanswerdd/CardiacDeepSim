"""Fig 4: normalised Vm on the 24 x 24 tissue at t = 160, 162, ..., 182 of the test set,
(a) LSTM-assisted simulation started at t = 160, (b) golden standard.

t is the step index in the test set; the absolute step in the data set is t + 43840. The colour scale is the
range of the z-scored test Vm (-1.0850763396420926 to 3.24444338332082).

    python3 05_figures/fig4_propagation.py
"""
import os
import sys

import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _style  # noqa: E402,F401
from _style import plt  # noqa: E402
from common import load_lstm, lstm_simulate, out_dir, test_data  # noqa: E402

start = 160
data = test_data()
norm = mcolors.Normalize(vmin=data[:, :, 0].min(), vmax=data[:, :, 0].max())
pred = lstm_simulate(load_lstm(), data, start, 24)
times = [start + 2 * t for t in range(12)]
grids = {"lstm": np.array([pred[t - start, :, 0].reshape(24, 24) for t in times]),
         "truth": np.array([data[t, :, 0].reshape(24, 24) for t in times])}

d = out_dir("05_figures")
np.savez_compressed(os.path.join(d, "fig4_propagation.npz"), t=np.array(times), absolute_step=np.array(times) + 43840,
                    vmin=norm.vmin, vmax=norm.vmax, **grids)
for name, label in [("lstm", "a_lstm"), ("truth", "b_golden_standard")]:
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(9, 3), subplot_kw={"xticks": [], "yticks": []})
    for t, ax, grid in zip(times, axs.flat, grids[name]):
        ax.imshow(norm(grid), interpolation="quadric", cmap="coolwarm")
        ax.set_title(f"t={t}")
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"fig4{label}.png"), dpi=500)
    plt.close(fig)
    print(f"wrote outputs/05_figures/fig4{label}.png")
