"""
Simple example of an analytics pipeline using the Doreisa framework and Dask arrays.

This script demonstrates how to:
- Initialize the Doreisa head node
- Define sliding-window array inputs for simulation data (e.g., fields "U" and "V")
- Register a callback function that is called at each timestep of the simulation
- Compute and report basic analytics (average values of V and U over time steps)

"""

import dask.array as da
from doreisa.head_node import init
from doreisa.window_api import ArrayDefinition, run_simulation


def simulation_callback(
    V: list[da.Array],
    U: list[da.Array],
    timestep: int
):
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
    Vavg = V[0].mean().compute()
    Uavg = U[0].mean().compute()

    # Print formatted analytics information for the current step
    print(f"[ANALYTICS] Average at timestep {
          timestep}: V={Vavg}, U={Uavg}", flush=True)


# --- Main execution section ---

# Initialize the Doreisa head node
# and registers this process as the analytics controller.
init()
print("Analytics Initialized", flush=True)


# Invoke the `simulation_callback` at each timestep
# Respect the configured `window_size` for each array
# Stop after `max_iterations` timesteps
run_simulation(
    simulation_callback,
    [
        ArrayDefinition("V", window_size=1),
        ArrayDefinition("U", window_size=1),
    ],
    max_iterations=10,
)
