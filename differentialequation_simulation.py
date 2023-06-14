# Import the cbcbeat module
import matplotlib.pyplot as plt
from cbcbeat import *
import numpy as np
import pickle

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 2

# Turn off adjoint functionality
import cbcbeat
if cbcbeat.dolfin_adjoint:
    parameters["adjoint"]["stop_annotating"] = True

# Define the computational domain
resolution = 100
mesh = UnitSquareMesh(resolution, resolution)
N = mesh.num_vertices()
tme = Constant(0.0)
dt = 1
cycle_len = 500
ndt = cycle_len * 110

# Define the conductivity (tensors)
M_i = 1./800
M_e = 0.

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = Tentusscher_panfilov_2006_epi_cell()

# Define some external stimulus
tol = 1E-14
durr = 1.
k_0 = 100.
k_1 = 0
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.5 + tol
class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.8 - tol and x[0] >= 0.8 - tol
stimulation_cells = MeshFunction('size_t', mesh, 2)
subdomain0 = Omega_0()
subdomain1 = Omega_1()
stimulation_cells.set_all(0)
subdomain1.mark(stimulation_cells, 1)

class K(UserExpression):
    def __init__(self, subdomains, k_0, k_1, t, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.k_0 = k_0
        self.k_1 = k_1
        self.t = t

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 1:
            if float(self.t) % cycle_len <= durr:
                values[0] = self.k_0
            else:
                values[0] = self.k_1
        else:
            values[0] = self.k_1

I_s = K(stimulation_cells, k_0, k_1, t=tme, degree=0)


# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, tme, M_i, M_e, cell_model, I_s)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps["theta"] = 0.5                        # Second order splitting scheme
ps["pde_solver"] = "monodomain"          # Use Monodomain model for the PDEs
ps["apply_stimulus_current_to_pde"] = True
ps["CardiacODESolver"]["scheme"] = "RL1" # 1st order Rush-Larsen for the ODEs
ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
ps["MonodomainSolver"]["algorithm"] = "cg"
ps["MonodomainSolver"]["theta"] = 1.     #theta
ps["MonodomainSolver"]["use_custom_preconditioner"] = True
ps["MonodomainSolver"]["preconditioner"] = "jacobi"

solver = SplittingSolver(cardiac_model, params=ps)

##############
#   dataset  #
##############
data = []
label = []

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())
collect_idx = [0, 13, 15, 16]

# Time stepping parameters
interval = (0.0, dt * ndt)

timer = Timer("XXX Forward solve") # Time the total solve

# Solve!
ti = 0
vvv = []
caiii = []
casrr = []
casss = []
for (timestep, (vs_, vs, vur)) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep[0], timestep[1])
    data.append([[[vs(i/resolution, j/resolution)[k] for k in collect_idx] for j in range(resolution)] for i in range(resolution)])

    if float(timestep[0]) % cycle_len == 0 and float(timestep[0]) != 0:
        data_out = np.array(data)
        with open(f'output_{int(timestep[0])}.pickle', 'wb') as handle:
            pickle.dump(data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ti += 1

timer.stop()

data = np.array(data)
with open('output.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)