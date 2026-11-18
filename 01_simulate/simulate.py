"""Step 01: generate the "golden standard" data with cbcbeat (Section 4.1 of the paper).

Monodomain model + ten Tusscher-Panfilov 2006 epicardial cell model on the unit square, dt = 1 ms, pacing cycle
500 ms, stimulus 100 (1 ms) in [0.8, 1] x [0.8, 1]. Vm, Cai, CaSR and Cass are sampled on a 24 x 24 grid at every
step and saved as a (steps, 24, 24, 4) float64 array. The paper's data set is the first 54801 steps
(about 109.6 cycles).

    python3 01_simulate/simulate.py                       # full run, 54801 steps
    python3 01_simulate/simulate.py --steps 200 --check   # first 200 steps, compared with the published data
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

from cbcbeat import *  # noqa: E402,F401,F403

# (common.py needs torch, which is not in the simulation image, so the few constants are repeated here)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.environ.get("PUBLISHED_DIR_IN_CONTAINER", os.path.join(ROOT, "published"))
PUBLISHED_DATA = "output0328-1.pickle"
N_STEPS = 54801

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=N_STEPS, help="number of 1 ms steps (paper: 54801)")
parser.add_argument("--check", action="store_true", help="compare the result with published/" + PUBLISHED_DATA)
args = parser.parse_args()

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 2

import cbcbeat  # noqa: E402

if cbcbeat.dolfin_adjoint:
    parameters["adjoint"]["stop_annotating"] = True

# Computational domain
resolution = 24
mesh = UnitSquareMesh(resolution, resolution)
tme = Constant(0.0)
dt = 1
cycle_len = 500

# Conductivity
M_i = 1. / 800
M_e = 0.

cell_model = Tentusscher_panfilov_2006_epi_cell()

# External stimulus: 100 for the first 1 ms of every cycle in [0.8, 1] x [0.8, 1]
tol = 1E-14
durr = 1.
k_0 = 100.
k_1 = 0


class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.8 - tol and x[0] >= 0.8 - tol


stimulation_cells = MeshFunction('size_t', mesh, 2)
stimulation_cells.set_all(0)
Omega_1().mark(stimulation_cells, 1)


class K(UserExpression):
    def __init__(self, subdomains, k_0, k_1, t, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.k_0 = k_0
        self.k_1 = k_1
        self.t = t

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 1 and float(self.t) % cycle_len <= durr:
            values[0] = self.k_0
        else:
            values[0] = self.k_1


I_s = K(stimulation_cells, k_0, k_1, t=tme, degree=0)
cardiac_model = CardiacModel(mesh, tme, M_i, M_e, cell_model, I_s)

ps = SplittingSolver.default_parameters()
ps["theta"] = 0.5                        # second order splitting
ps["pde_solver"] = "monodomain"
ps["apply_stimulus_current_to_pde"] = True
ps["CardiacODESolver"]["scheme"] = "RL1"  # first order Rush-Larsen
ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
ps["MonodomainSolver"]["algorithm"] = "cg"
ps["MonodomainSolver"]["theta"] = 1.
ps["MonodomainSolver"]["use_custom_preconditioner"] = True
ps["MonodomainSolver"]["preconditioner"] = "jacobi"
solver = SplittingSolver(cardiac_model, params=ps)

(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())
collect_idx = [0, 13, 15, 16]  # V, Cai, CaSR, Cass in the state vector of the TP06 model

data = np.empty((args.steps, resolution, resolution, len(collect_idx)))
t0 = time.time()
for n, (timestep, (vs_, vs, vur)) in enumerate(solver.solve((0.0, dt * args.steps), dt)):
    for i in range(resolution):
        for j in range(resolution):
            data[n, i, j] = np.asarray(vs(i / resolution, j / resolution))[collect_idx]
    if (n + 1) % cycle_len == 0:
        print(f"step {n + 1}/{args.steps}  ({time.time() - t0:.0f} s)", flush=True)
    if n + 1 == args.steps:
        break

os.makedirs(os.path.join(ROOT, "outputs", "01_simulate"), exist_ok=True)
# only a full run is written as output.pickle (the input of step 02); shorter runs get their own name
name = "output.pickle" if args.steps == N_STEPS else f"output_{args.steps}steps.pickle"
out = os.path.join(ROOT, "outputs", "01_simulate", name)
with open(out, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"wrote outputs/01_simulate/{name} {data.shape} in {time.time() - t0:.0f} s")

if args.check:
    with open(os.path.join(PUBLISHED, PUBLISHED_DATA), "rb") as handle:
        ref = pickle.load(handle)[:args.steps]
    diff = np.abs(data - ref).max()
    print(f"max |difference| to published/{PUBLISHED_DATA} over the first {args.steps} steps: {diff:.3e}")
    if diff > 1e-6:
        raise SystemExit("CHECK FAILED")
    print("CHECK PASSED")
