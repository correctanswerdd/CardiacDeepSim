# CardiacDeepSim

论文：W. Dong, Y. Li, R. Zhu, "Deep-Learning-Assisted Cardiac Electrophysiology Simulation," *2023 Winter Simulation Conference (WSC)*, San Antonio, TX, 2023, pp. 758-769. https://doi.org/10.1109/WSC60868.2023.10408359

## 1. 安装环境

- Docker（Windows 用 Docker Desktop）。
- bash（Windows 用 Git Bash 或 WSL）。

第一次运行 `./run.sh` 时用 `Dockerfile` 和 `requirements.txt` 构建镜像。

## 2. 按顺序的命令

```bash
./run.sh make-data
./run.sh evaluate table1
./run.sh figures fig1_traces
./run.sh figures fig2_vm_comparison
./run.sh figures fig3_long_term
./run.sh figures fig4_propagation
```

## 3. 论文里每一项对应的脚本和输出

| 论文 | 脚本 | 输出 |
|---|---|---|
| Table 1 | `04_evaluate/table1.py` | `outputs/04_evaluate/table1.json`、`table1.md` |
| Fig 1 | 流程图用 PPT 直接画；图里的 4 条波形：`05_figures/fig1_traces.py` | `outputs/05_figures/fig1_trace_{Vm,Cai,CaSR,Cass}.png` |
| Fig 2 | `05_figures/fig2_vm_comparison.py` | `outputs/05_figures/fig2_vm_comparison.png` |
| Fig 3 | `05_figures/fig3_long_term.py` | `outputs/05_figures/fig3_long_term.png` |
| Fig 4 (a)、(b) | `05_figures/fig4_propagation.py` | `outputs/05_figures/fig4a_lstm.png`、`fig4b_golden_standard.png` |
