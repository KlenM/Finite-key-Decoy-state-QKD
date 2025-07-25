# Finite-key Decoy-State Quantum Key Distribution Protocol

Implements a finite-key decoy-state protocol for quantum key distribution (QKD) analysis.


## Install

```bash
pip install git+https://github.com/KlenM/Finite-key-Decoy-state-QKD.git
```

Or clone the repository and run `pip install .` from the project folder.

## Usage

```python
from decoy_state import DecoyStateFinite

params = dict(
    e_mis = 3.3e-2, # error rate due to optical errors
    f_EC = 1.22, # error correction inefficiency
    Y_0 = 1.7e-6, # dark counts
    eps_sec = 1e-10,
    eps_cor = 1e-15,
)
protocol = DecoyStateFinite.from_channel_params_optimize(
    transmittance=5e-4, N=1e10, **params
)
print(protocol)
print(f" Key length = {int(protocol.key_length())}")
```

```
DecoyStateFinite(mu_k=[0.43, 0.18, 0.00], p_k=[0.14, 0.51, 0.35], n_Xk=[1.5e+05, 2.3e+05, 3.0e+03], n_Zk=[2.5e+04, 3.8e+04, 4.9e+02], m_Xk=[5.6e+03, 9.8e+03, 1.5e+03], m_Zk=[9.0e+02, 1.6e+03, 2.4e+02], f_EC=1.22, eps_sec=1.0e-10, eps_cor=1.0e-15)
Key length = 41368
```
