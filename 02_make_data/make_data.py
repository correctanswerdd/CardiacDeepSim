"""Step 02: turn the simulation output into the data set used by steps 03-05.

Reads the (54801, 24, 24, 4) simulation array, flattens the grid to 576 cells (cell = row * 24 + column) and
checks the 80/20 split lengths and the z-scored Vm range of the data set. The z-scoring and the 80/20 split
are done where the data is used (common.py); this step only stores the flattened array.

    python3 02_make_data/make_data.py                       # from outputs/01_simulate/output.pickle
    python3 02_make_data/make_data.py --source published    # from published/output0328-1.pickle
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (DATASET, FINGERPRINT, N_ATTR, N_CELLS, N_STEPS, OUTPUTS, PUBLISHED, PUBLISHED_DATA,  # noqa: E402
                    PUBLISHED_DATA_SHA256, out_dir, preprocess, sha256, split, write_json)

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["simulation", "published"], default="simulation")
args = parser.parse_args()

if args.source == "published":
    src = os.path.join(PUBLISHED, PUBLISHED_DATA)
    if not os.path.isfile(src):
        raise SystemExit(f"{src} is missing: copy {PUBLISHED_DATA} into published/ (see README)")
    digest = sha256(src)
    print("sha256", digest)
    if digest != PUBLISHED_DATA_SHA256:
        raise SystemExit(f"sha256 of {PUBLISHED_DATA} differs from the published file")
else:
    src = os.path.join(OUTPUTS, "01_simulate", "output.pickle")
    if not os.path.isfile(src):
        raise SystemExit("outputs/01_simulate/output.pickle is missing: run step 01, or use --source published")
    digest = sha256(src)

with open(src, "rb") as handle:
    raw = np.asarray(pickle.load(handle))
print("loaded", src, raw.shape, raw.dtype)
if raw.shape != (N_STEPS, 24, 24, N_ATTR):
    raise SystemExit(f"expected shape {(N_STEPS, 24, 24, N_ATTR)}")
data = raw.reshape(N_STEPS, N_CELLS, N_ATTR)

# data check
train, test = split(data)
full_z = preprocess(data.copy())[:, :, 0]
test_z = preprocess(test.copy())[:, :, 0]
found = {
    "n_train": train.shape[0],
    "n_test": test.shape[0],
    "full_vm_zscore_min": float(full_z.min()),
    "full_vm_zscore_max": float(full_z.max()),
    "test_vm_zscore_min": float(test_z.min()),
    "test_vm_zscore_max": float(test_z.max()),
}
ok = all(abs(found[k] - v) < 1e-9 for k, v in FINGERPRINT.items())

d = out_dir("02_make_data")
np.save(DATASET, data)
write_json(os.path.join(d, "dataset.json"), {
    "source": os.path.relpath(src, os.path.dirname(OUTPUTS)), "source_sha256": digest,
    "shape": list(data.shape), "attributes": ["Vm", "Cai", "CaSR", "Cass"],
    "check": found})
print("wrote outputs/02_make_data/dataset.npy")
if not ok:
    raise SystemExit("DATA CHECK FAILED: " + str(found))
print("data check passed")
