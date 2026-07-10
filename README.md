# Paper code

Code accompanying [*Axion Misalignment Across First-Order Phase Transitions*](https://arxiv.org/abs/2607.01333).

This repository contains two independent packages:

| Package | Description |
|---------|-------------|
| [`sim_core/`](sim_core/) | Lattice simulation of axion dynamics during a first-order phase transition |
| [`xi_model/`](xi_model/) | Compact inference model for the axion dark matter abundance ratio ξ |

## Repository layout

```
paper_code/
├── sim_core/
│   ├── axion_sim.py      # core simulation engine
│   ├── run_sweep.py      # parameter sweep driver
│   ├── requirements.txt
│   └── README.md
└── xi_model/
    ├── xi_model/
    │   ├── api.py        # XiModel class and predict()
    │   └── cli.py
    ├── data/             # pre-computed geometry bank and fit tables
    ├── examples/
    ├── tests/
    ├── pyproject.toml
    └── README.md
```

## Quick start

### Simulation (`sim_core`)

```bash
pip install numpy numba pyfftw scipy matplotlib
# macOS: brew install fftw   |   Linux: apt install libfftw3-dev
python sim_core/run_sweep.py
```

### ξ model (`xi_model`)

```bash
pip install -e xi_model/
python -c "
from xi_model import XiModel
m = XiModel()
r = m.predict(hstar=0.7, vw=0.6, theta0=1.0, beta_over_h=8.0)
print(f'xi = {r.xi:.4f}')
"
```

or from the command line:

```bash
python -m xi_model --hstar 0.7 --vw 0.6 --theta0 1.0 --betaH 8.0 --pretty
```

## Citation

If you use this code, please cite [*Axion Misalignment Across First-Order Phase Transitions*](https://arxiv.org/abs/2607.01333).
