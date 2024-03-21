import numpy as np
import matplotlib

matplotlib.rcParams["font.size"] = 8
import matplotlib.pyplot as plt
from functools import partial
import jax, optax
import jax.lax as lax
import jax.numpy as jnp
from mesogif_exp_jax import (
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
from scipy.signal import welch

tau_syn = 3.0
J = 10.0
rate = 6562.5
mu_ext = (
    tau_syn * rate / 1000.0 * J
)  # divide by factor 1000. because rate is given in s^-1 instead of ms^-1
sigma2_ext = tau_syn * rate / 1000.0 * J ** 2

sigma_noise = np.sqrt(sigma2_ext / 2.0)  # See Eq. 23 in note

###############
## Simulate
###############

steps = 10000
M = 2
K = 500
synport = np.array([[0], [1]], dtype=np.int32)
staticparams = StaticParams(
    synport=synport,
    u_reset=np.array(0.0, dtype=np.float32),
    sigma_noise=sigma_noise,
    tau_noise=tau_syn,
    dt=0.5,
    num_steps=2000,
    num_ref=4,
    delay=1.0,
)

w = np.ones((M, 2), dtype=jnp.float32)
w[:, 0] = 10.0 * 0.1 * 0.0
w[:, 1] = -50.0 * 0.1 * 0.0
tau_s = np.ones((M, 2), dtype=np.float32)
tau_s[:, 0] = 3.0
tau_s[:, 1] = 3.0
N = np.zeros(2, dtype=np.float32)
N[0] = 800
N[1] = 200
params = Params(
    RI=jnp.ones(M, dtype=jnp.float32) * 15.75,
    tau_m=jnp.ones(M, dtype=jnp.float32) * 20.0,
    N=jnp.array(N, dtype=jnp.float32),
    tau_s=jnp.array(tau_s),
    u_thr=jnp.ones(M, dtype=jnp.float32) * 15.0,
    C_mem=jnp.ones(M, dtype=jnp.float32) * 250.0,
    w=jnp.array(w),
)

rec = simulate(
    params,
    staticparams,
    K,
    steps,
    record=["u", "h", "y", "spikes", "m", "x", "lambd_old"],
)

histograms = rec["spikes"].T
fs, psd = welch(histograms, fs=1000)
fig, ax = plt.subplots(2, 2, sharex="row", sharey="row")
ax[0, 0].plot(histograms[0])
ax[0, 1].plot(histograms[1])
ax[1, 0].plot(fs, psd[0])
ax[1, 1].plot(fs, psd[1])
ax[0, 0].set_xlabel("t (ms)")
ax[0, 1].set_xlabel("t (ms)")
ax[1, 0].set_xlabel("f (Hz)")
ax[1, 1].set_xlabel("f (Hz)")
fig.tight_layout()
plt.show()
