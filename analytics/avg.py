"""
Simple example of an analytics pipeline using the deisa-ray framework and Dask arrays.

This script demonstrates how to:
- Initialize the deisa-ray head node
- Define sliding-window array inputs for simulation data (e.g., fields "U" and "V")
- Register a callback function that is called at each timestep of the simulation
- Compute and report basic analytics (average values of V and U over time steps)

"""

from deisa.ray import Deisa
from deisa.ray.types import DeisaArray

d = Deisa()


@d.register("U", "V")
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
    Vavg = V[0].mean().compute()
    Uavg = U[0].mean().compute()

    # Print formatted analytics information for the current step
    print(f"[ANALYTICS] Average at timestep {U[0].t}: V={Vavg}, U={Uavg}", flush=True)


# --- Main execution section ---

# Initialize the deisa-ray head node
# and registers this process as the analytics controller.

print("Analytics Initialized", flush=True)

# Invoke the `simulation_callback` at each timestep
d.execute_callbacks()
