# CardiacDeepSim

论文：W. Dong, Y. Li, R. Zhu, "Deep-Learning-Assisted Cardiac Electrophysiology Simulation," *2023 Winter Simulation Conference (WSC)*, San Antonio, TX, 2023, pp. 758-769. https://doi.org/10.1109/WSC60868.2023.10408359

这个文件夹按步骤编号，每一步读上一步的输出，写到 `outputs/` 下自己的子文件夹：

```
01_simulate/    cbcbeat 仿真（单域模型 + ten Tusscher-Panfilov 2006）    -> outputs/01_simulate
02_make_data/   把仿真结果整理成数据集                                    -> outputs/02_make_data
03_train/       训练 LSTM 和 Neural ODE                                   -> outputs/03_train
04_evaluate/    Table 1                                                   -> outputs/04_evaluate
05_figures/     Fig 1（里面的波形）、Fig 2、Fig 3、Fig 4                   -> outputs/05_figures
checkpoints/    论文的两个模型：LSTMbest.pt、NeuralODE.pt
published/      论文的数据
```

## 1. 安装环境

- Docker（Windows 用 Docker Desktop）。
- 一个 bash：Linux/macOS 自带；Windows 用 Git Bash 或 WSL。
- 第 3 步（训练）用 NVIDIA GPU（Docker 要能用 `--gpus all`）。

不用手动装任何 Python 包。第一次运行 `./run.sh` 时会自动构建两个镜像：

| 镜像 | 文件 | 内容 | 用在 |
|---|---|---|---|
| `cds-sim:2026-11` | `Dockerfile.sim` | simcardems 官方镜像（FEniCS + cbcbeat） | 第 1 步 |
| `cds-torch:2026-11` | `Dockerfile` + `requirements.txt` | PyTorch 官方镜像 + torchdiffeq、matplotlib、seaborn | 第 2 到 5 步 |

### 数据和模型

```
published/output0328-1.pickle   论文的数据：54801 步 × 24 × 24 个网格点 × 4 个量（Vm、Cai、CaSR、Cass）
checkpoints/LSTMbest.pt         LSTM-assisted simulation
checkpoints/NeuralODE.pt        Neural ODE
```

## 2. 按顺序的命令

在这个文件夹里打开 bash。

### A. 用论文的数据和模型算出论文的表和图

```bash
./run.sh paper
```

等于下面几条：

```bash
./run.sh make-data --source published
./run.sh evaluate table1
./run.sh figures fig1_traces
./run.sh figures fig2_vm_comparison
./run.sh figures fig3_long_term
./run.sh figures fig4_propagation
```

### B. 从头跑

```bash
./run.sh simulate --check
./run.sh make-data
./run.sh train lstm
./run.sh train node
./run.sh evaluate table1 --checkpoints outputs/03_train
```

第 4、5 步默认用 `checkpoints/` 里的模型；加 `--checkpoints outputs/03_train` 改用第 3 步训练的模型。

### 每一步

**第 1 步 `01_simulate/simulate.py`**：单位正方形，单域模型 + ten Tusscher-Panfilov 2006 心外膜细胞模型，dt = 1 ms，周期 500 ms，每个周期开头在 [0.8, 1] × [0.8, 1] 加 1 ms、强度 100 的刺激。每一步在 24 × 24 个点上取 Vm、Cai、CaSR、Cass，共 54801 步，写到 `outputs/01_simulate/output.pickle`。`--check` 和 `published/` 里的数据比对；`--steps 500 --check` 只跑前 500 步。

**第 2 步 `02_make_data/make_data.py`**：把网格拉平成 576 个点（点号 = 行 × 24 + 列），写 `outputs/02_make_data/dataset.npy`。默认读第 1 步的结果，`--source published` 读 `published/` 里的数据。前 80% 的步是训练集，后 20% 是测试集。

**第 3 步 `03_train/`**：`train_lstm.py`（Algorithm 1）和 `train_node.py`，超参数同论文 4.3 节：Adam，LSTM 学习率 1e-3、batch 16、2 层（hidden 64）、3 个 epoch；Neural ODE 学习率 0.01、2 个 epoch；`--seed 0`。

**第 4 步 `04_evaluate/table1.py`**：两个模型从测试集第 2067 步开始模拟一个周期（500 步），按式 (9) 在这 500 步和 576 个点上算 MSE，每个量分别缩放到 [-1, 1]。Neural ODE 的 z0 采样用种子 0。

**第 5 步 `05_figures/`**：每张图一个脚本，同时存画图用的数值（`.npz`）。

## 论文里每一项对应的脚本和输出

| 论文 | 脚本 | 输出 |
|---|---|---|
| Table 1（MSE） | `04_evaluate/table1.py` | `outputs/04_evaluate/table1.json`、`table1.md` |
| Fig 1 | 流程图用 PPT 直接画；图里的 4 条波形：`05_figures/fig1_traces.py` | `outputs/05_figures/fig1_trace_{Vm,Cai,CaSR,Cass}.png` |
| Fig 2 | `05_figures/fig2_vm_comparison.py` | `outputs/05_figures/fig2_vm_comparison.png` |
| Fig 3 | `05_figures/fig3_long_term.py` | `outputs/05_figures/fig3_long_term.png` |
| Fig 4 (a)、(b) | `05_figures/fig4_propagation.py` | `outputs/05_figures/fig4a_lstm.png`、`fig4b_golden_standard.png` |

## 文件

```
run.sh                        所有命令的入口
Dockerfile.sim                第 1 步的镜像
Dockerfile, requirements.txt  第 2 到 5 步的镜像
common.py                     路径、数据、归一化、两个模型和它们的模拟
01_simulate/simulate.py
02_make_data/make_data.py
03_train/train_lstm.py, train_node.py
04_evaluate/table1.py
05_figures/fig1_traces.py, fig2_vm_comparison.py, fig3_long_term.py, fig4_propagation.py, _style.py
checkpoints/                  论文的两个模型
published/                    论文的数据
outputs/                      运行结果
```
