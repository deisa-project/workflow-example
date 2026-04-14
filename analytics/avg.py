"""
Simple example of an analytics pipeline using the deisa-ray framework and Dask arrays.

This script demonstrates how to:
- Initialize the deisa-ray head node
- Define sliding-window array inputs for simulation data (e.g., fields "U" and "V")
- Register a callback function that is called at each timestep of the simulation
- Compute and report basic analytics (average values of V and U over time steps)

"""

import dask.array as da
import deisa.ray as deisa
from deisa.ray.types import DeisaArray, WindowSpec
from deisa.ray.window_handler import Deisa


def simulation_callback(V: list[DeisaArray], U: list[DeisaArray]):
    """
    Callback function invoked by the simulation at each timestep.

    Parameters
    ----------
    V : list[dask.array.Array]
        A list of Dask arrays representing the V field values.
    U : list[dask.array.Array]
        A list of Dask arrays representing the U field values.
    timestep : int
        The current simulation timestep for which this callback is invoked.

    Notes
    -----
    This function performs a simple example analysis: computing the mean
    value of both fields `U` and `V` using Dask's lazy computation model.
    The `.compute()` call forces execution across the Dask cluster or
    local threads/processes.

    """
    # Compute the average
    Vavg = V[0].dask.mean().compute()
    Uavg = U[0].dask.mean().compute()

    # Print formatted analytics information for the current step
    print(f"[ANALYTICS] Average at timestep {U[0].t}: V={Vavg}, U={Uavg}", flush=True)


# --- Main execution section ---

# Initialize the deisa-ray head node
# and registers this process as the analytics controller.
deisa.config.enable_experimental_distributed_scheduling(True)
d = Deisa()

print("Analytics Initialized", flush=True)


# Invoke the `simulation_callback` at each timestep
# Respect the configured `window_size` for each array
# Stop after `max_iterations` timesteps
d.register_callback(
    simulation_callback,
    [
        WindowSpec("V"),
        WindowSpec("U"),
    ],
)
d.execute_callbacks()
