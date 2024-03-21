import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import jax, optax
import jax.lax as lax
import jax.numpy as jnp
from mesogif_jax import (
    Params,
    StaticParams,
    State,
    integrate_log_prob,
    update_state,
    compute_current_propagators,
    compute_current_voltage_propagators,
    compute_voltage_propagators,
    simulate,
)
import pickle

steps = 2000
M = 2
K = 1000
synport = np.array([[0], [1]], dtype=np.int32)
staticparams = StaticParams(
    synport=synport,
    u_reset=np.array(0.0, dtype=np.float32),
    dt=0.5,
    num_ref=4,
    delay=2.0,
)

w = np.ones((M, 2), dtype=jnp.float32)
w[:, 0] = 13.0 * np.exp(1) * 0.1
w[:, 1] = -45.0 * np.exp(1) * 0.1
tau_s = np.ones((M, 2), dtype=np.float32)
tau_s[:, 0] = 2.0
tau_s[:, 1] = 3.0
N = np.zeros(2, dtype=np.float32)
N[0] = 800
N[1] = 200
params = Params(
    RI=jnp.ones(M, dtype=jnp.float32) * 25.0,
    tau_m=jnp.ones(M, dtype=jnp.float32) * 17.0,
    N=jnp.array(N),
    tau_s=jnp.array(tau_s),
    u_thr=jnp.ones(M, dtype=jnp.float32) * 22.0,
    c=jnp.ones(M, dtype=jnp.float32) * 0.0,
    delta_u=jnp.ones(M, dtype=jnp.float32) * 0.8,
    C_mem=jnp.ones(M, dtype=jnp.float32) * 250.0,
    w=jnp.array(w),
)

params = jax.tree_util.tree_map(jnp.array, params)
rec = simulate(
    params,
    staticparams,
    K,
    steps,
    record=["u", "h", "y", "spikes", "m", "x", "lambd_old"],
)
