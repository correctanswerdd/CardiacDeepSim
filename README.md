# CardiacDeepSim

论文：W. Dong, Y. Li, R. Zhu, "Deep-Learning-Assisted Cardiac Electrophysiology Simulation," *2023 Winter Simulation Conference (WSC)*, San Antonio, TX, 2023, pp. 758-769. https://doi.org/10.1109/WSC60868.2023.10408359

```
01_simulate/    cbcbeat 仿真                    -> outputs/01_simulate
02_make_data/   数据集                          -> outputs/02_make_data
03_train/       训练 LSTM 和 Neural ODE         -> outputs/03_train
04_evaluate/    Table 1                         -> outputs/04_evaluate
05_figures/     Fig 1（波形）、Fig 2、Fig 3、Fig 4 -> outputs/05_figures
checkpoints/    LSTMbest.pt、NeuralODE.pt
published/      output0328-1.pickle（仿真数据）
```

## 1. 安装环境

- Docker（Windows 用 Docker Desktop）。
- bash（Windows 用 Git Bash 或 WSL）。
- 第 3 步用 NVIDIA GPU（Docker 要能用 `--gpus all`）。

第一次运行 `./run.sh` 时构建两个镜像：

| 镜像 | 文件 | 用在 |
|---|---|---|
| `cds-sim:2026-11` | `Dockerfile.sim` | 第 1 步 |
| `cds-torch:2026-11` | `Dockerfile` + `requirements.txt` | 第 2 到 5 步 |

## 2. 按顺序的命令

### A. 用论文的数据和模型

```bash
./run.sh make-data --source published
./run.sh evaluate table1
./run.sh figures fig1_traces
./run.sh figures fig2_vm_comparison
./run.sh figures fig3_long_term
./run.sh figures fig4_propagation
```

（`./run.sh paper` 依次运行以上命令。）

### B. 从头跑

```bash
./run.sh simulate
./run.sh make-data
./run.sh train lstm
./run.sh train node
./run.sh evaluate table1 --checkpoints outputs/03_train
```

## 3. 论文里每一项对应的脚本和输出

| 论文 | 脚本 | 输出 |
|---|---|---|
| Table 1 | `04_evaluate/table1.py` | `outputs/04_evaluate/table1.json`、`table1.md` |
| Fig 1 | 流程图用 PPT 直接画；图里的 4 条波形：`05_figures/fig1_traces.py` | `outputs/05_figures/fig1_trace_{Vm,Cai,CaSR,Cass}.png` |
| Fig 2 | `05_figures/fig2_vm_comparison.py` | `outputs/05_figures/fig2_vm_comparison.png` |
| Fig 3 | `05_figures/fig3_long_term.py` | `outputs/05_figures/fig3_long_term.png` |
| Fig 4 (a)、(b) | `05_figures/fig4_propagation.py` | `outputs/05_figures/fig4a_lstm.png`、`fig4b_golden_standard.png` |
