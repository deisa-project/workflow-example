# Example Workflow For InSitu Data Analytics
This repository contains a simulation based on the **Gray–Scott model** implemented in **C**, **C++**, and **Python**.  
It demonstrates integration of **Doreisa** for in-situ analytics with CPU and GPU simulation.  
The repository includes standalone and Doreisa-enabled variants, as well as ready-to-use launch scripts for each configuration.



---
## ⚙️ Simulation Variants

| Version | Language | Features | File(s) |
|----------|-----------|-----------|----------|
| **C (CPU)** | C | Basic CPU implementation with MPI | `c/src/sim.c` |
| **C++ + Doreisa** | C++20 | Adds Doreisa in-situ analytics support | `cpp/cpu/sim_doreisa.cpp` |
| **C++ + Kokkos (GPU)** | C++20 | GPU acceleration using Kokkos | `cpp/gpu/sim_kokkos-doreisa.cpp` |
| **Python (CPU)** | Python 3 | CPU simulation using NumPy | `python/sim.py` |
| **Python + Doreisa** | Python 3 | Adds in-situ analytics via Doreisa | `python/sim-doreisa.py` |

---

## 📁 Project Structure

```
.
├── analytics/
│   └── avg.py                # Simple in-situ analytics example: computes average of U and V
│
├── c/
│   └── src/
│       └── sim.c             # Pure C implementation (CPU only, no Doreisa)
├── cpp/
│   ├── cpu/                  # C++ implementation (CPU only, Doreisa enabled)
│   │   ├── CMakeLists.txt    
│   │   └── sim-doreisa.cpp
│   └── gpu/                  # C++ implementation (GPU, Doreisa enabled)
│       ├── CMakeLists.txt
│       └── sim-kokkos-doreisa.cpp
├── nix/
├── python/
│   ├── requirements.txt      # Python dependencies for the simulation and analytics
│   ├── sim.py                # Python simulation
│   └── sim-doreisa.py        # Python simulation integrated with Doreisa
│
├── launch-scripts/
│   ├── launch-scripts.sh     # TODO
│   └── ...
├── Makefile                  # Optional build helper for C targets
└── README.md                 # This file
```

---

## 📊 In-Situ Analytics (`analytics/`)

The **analytics** processes simulation data as it runs using **Doreisa’s API**.

### Example: `avg.py`
```python
def simulation_callback(V, U, timestep):
    Vavg = V[0].mean().compute()
    Uavg = U[0].mean().compute()
    print(f"[ANALYTICS] Average at timestep {timestep}: V={Vavg}, U={Uavg}")
```

This function receives Dask-arrays from the simulation and computes average values at each iteration.

---

## 🚀 Running the Simulations

All versions can be launched using the provided shell scripts in `launch-scripts/`.


### 🧩 Example
```
bash launch-scripts/launch-insitu-cpp-kokkos-local.sh
```
---
### 📈 Output

During execution, the analytics prints progress and timing information like:
```
Analytics Initialized
[SIM, rank 1] connected to doreisa client
[SIM, rank 0] connected to doreisa client
[step      0] ranks=2 grid=2x1 N=512x256 local=256x256 Vsum=1.271578e+05 elapsed=0.02s GSS_time=1.38ms halo_time=0.06ms
[ANALYTICS] Average at timestep 0: V=0.01893053523662347, U=0.9701366492501763
[step      1] ranks=2 grid=2x1 N=512x256 local=256x256 Vsum=1.272115e+05 elapsed=0.04s GSS_time=1.24ms halo_time=0.06ms
[ANALYTICS] Average at timestep 1: V=0.018002955605113495, U=0.9705465046278077
[step      2] ranks=2 grid=2x1 N=512x256 local=256x256 Vsum=1.272645e+05 elapsed=0.06s GSS_time=1.14ms halo_time=0.05ms
[ANALYTICS] Average at timestep 2: V=0.017169531188245785, U=0.970950778183183
[step      3] ranks=2 grid=2x1 N=512x256 local=256x256 Vsum=1.273159e+05 elapsed=1.28s GSS_time=1.12ms halo_time=0.04ms
[ANALYTICS] Average at timestep 3: V=0.016425385413415462, U=0.9713431944620565
...
```
---

## 🧠 Dependencies

### For Python simulations
```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### For C/C++ builds
- **CMake ≥ 3.18**
- **MPI** (OpenMPI or MPICH)
- **Kokkos** (for GPU builds)
- **pybind11** (for Python integration)
- **CUDA Toolkit** (for GPU builds, optional)

---
